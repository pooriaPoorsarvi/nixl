from typing import Tuple, Any
import argparse
import torch
from nixl import nixl_agent, nixl_agent_config
from nixl.logging import get_logger
import time
import gc
import random
import statistics

logger = get_logger(__name__)

dtype: torch.dtype = torch.float32


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, required=True)
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--use_cuda", type=bool, default=True)
    parser.add_argument(
        "--mode",
        type=str,
        default="initiator",
        help="Local IP in target, peer IP (target's) in initiator",
    )
    parser.add_argument("--batch", type=int, default=1000)
    return parser.parse_args()

TARG = "target"
INIT = "initiator"
DONE_R = "Done_reading"
DONE = "DONE"

def get_message(one_call: bool, ind: int) -> str:
    '''
        one_call: whether we are using initialize_xfer or make_prepped_xfer
    '''
    return f"{DONE_R}_one_call_{one_call}_ind_{ind}"

def setup_agent(args):
    listen_port = args.port
    if args.mode == INIT:
        listen_port = 0
    
    config = nixl_agent_config(True, True, listen_port=int(listen_port))
    agent = nixl_agent(args.mode, config)

    return agent

def setup_memory(args, agent: nixl_agent, dims: tuple, dtype: torch.dtype) -> Tuple[torch.Tensor, Any]:
    '''
        setsup memory, returns tensors and its registers
    '''
    dev = "cpu"
    if args.use_cuda:
        dev = "cuda:0"
    torch.set_default_device(dev)

    init_f = torch.zeros
    if args.mode == TARG:
        init_f = torch.ones
    
    tensor: torch.Tensor = init_f(dims, dtype=dtype)
    reg_descs = agent.register_memory(tensor)
    if not reg_descs:
        raise RuntimeError("Couldn't register memory")
    
    return tensor, reg_descs

def get_dims(mi) -> Tuple:
    return [256 * (2 ** (mi)), 16]


def run_server(args, mi: int, agent: nixl_agent = None, row_ids = None, iterations: int = 1):
    """
        selects rows
        creates local xfer descs
        waits for other nodes metadata
        notifies other nodes of its local xfer descs so it can pull it as serialized value (we never send them our metadata, we just recieve it)
        waits for done message encoded in notifs of other agent if those notifs exist
        waits (blocks for targets to all have exited and invalidated)
        deregister
            release handles which we don't have
            deregisters memory
            deregisters tensor
            deregisters agent
        returns timing

    """
    if agent is None:
        agent = setup_agent(args)

    dims = get_dims(mi)
    tensor, reg_descs = setup_memory(args, agent, dims, dtype)
    logger.info("setup server")

    # doing it before waiting for other agent, so to not waste time 
    if row_ids is None:
        row_ids = [0]
        rows = [tensor]
    else:
        rows = [tensor[i] for i in row_ids]
    descs = agent.get_xfer_descs(rows)
    serialized_descs = agent.get_serialized_descs(descs)
    logger.info("serialized descs")

    while not agent.check_remote_metadata(INIT):
        continue

    agent.send_notif(INIT, serialized_descs)
    
    logger.info("waiting for client to finish")
    # Wait for the other agent to drop off
    while agent.check_remote_metadata(INIT): 
        continue

    # No one waiting on us and we are ready to remove all data

    logger.info("clearning up")
    # No handle here to remove
    agent.deregister_memory(reg_descs)
    del agent
    del tensor
    
    return





def run_client(args, mi: int, agent: nixl_agent = None, row_ids = None, iterations: int = 1, sample_size: int = 1000, batch: int = 1):
    """
        sends metadata to server
        pulls for server metadata info
        wait until serialized descs are recieved
        get the remote xfer descs of target
        desc target descs
        create local rows
        create local xfer descs
        if len(row_ids) == 1:
            initialize xfer:
                op,
                local xfer descs,
                remote xfer descs,
                remote agent name
                message to notify when done
        else:
            one time:
                prep xfer dlist for validation:
                    name : empty or target
                    descs
                make preped xfer:
                    op,
                    local prep dlist <> descs,
                    indicies,
                    remote prep dlist <> descs,
                    indicies,
                    not message
        both give a handle
        handle is not transfered yet so we need to transfer which returns and initial status
        immidiately check that the return is not err
        a loop of polling for status:
            check xfer state using handle
            if err exit
            if "DONE" finished
        
        deregister:
            release handles
            del descs and dlists
            deregister memory
            invalidate local metadata due to asking for TARG
            deregister agent
            then remove tensor

        return timing
    """

    if agent is None:    
        agent = setup_agent(args)

    dims = get_dims(mi)
    tensor, reg_descs = setup_memory(args, agent, dims, dtype)
    
    logger.info("client got data and descs")


    agent.send_local_metadata(args.ip, args.port)
    agent.fetch_remote_metadata(TARG, args.ip, args.port)

    # create local information before polling not to waste time
    if row_ids is None:
        row_ids = [0]
        rows = [tensor]
    else:
        rows = [tensor[i] for i in row_ids]
    local_desc = agent.get_xfer_descs(rows)

    
    logger.info("waiting for server metadata")
    while not agent.check_remote_metadata(TARG):
        continue

    logger.info("waiting for server serialized info")
    messages = agent.get_new_notifs()
    while len(messages) == 0:
        messages = agent.get_new_notifs()


    assert TARG in messages and len(messages[TARG]) > 0

    target_descs = agent.deserialize_descs(messages[TARG][0])

    number_of_calls = len(row_ids)
    if number_of_calls != 1:
        local_dlist = agent.prep_xfer_dlist("", local_desc)
        target_dlist = agent.prep_xfer_dlist(TARG, target_descs)

    handles = []
    for k in range(batch):
        if number_of_calls == 1:
            handles += [agent.initialize_xfer("READ", local_desc, target_descs, TARG)]
        else:
            idx = [k % number_of_calls]
            handles += [agent.make_prepped_xfer("READ", local_dlist, idx, target_dlist, idx)]

    results = []
    i = 0
    while i < iterations:
        b = min(batch, iterations - i)
        t0 = time.perf_counter()
        for j in range(b):
            if agent.transfer(handles[j]) == "ERR":
                raise RuntimeError("transfer errored out")
        for j in range(b):
            status = agent.check_xfer_state(handles[j])
            while status != DONE:
                status = agent.check_xfer_state(handles[j])
                if status == "ERR":
                    raise RuntimeError("transfer errored out")
        results += [(time.perf_counter() - t0) * 1000 / b]
        i += b

    for handle in handles:
        agent.release_xfer_handle(handle)

    # Read is done
    logger.info("cleaning up")
    agent.deregister_memory(reg_descs)
    # Invalidate so that the other agent knows we are off, should be done at the end so we leave everything else first
    del tensor
    agent.remove_remote_agent(TARG)
    agent.invalidate_local_metadata(args.ip, args.port)
    del agent

    return results


if __name__ == "__main__":
    args = parse_args()
    
    # agent = setup_agent(args)
    agent = None
    mi = 22
    warmup = 100
    iteration = 100000
    batch = args.batch
    dims = get_dims(mi)
    row_ids = [i for i in range(min(dims[0], 100000))]


    if args.mode == TARG:
        run_server(args, mi, agent, row_ids, iteration)
    else:
        gc.disable()
        res = run_client(args, mi, agent, row_ids, iteration, 1, batch)
        res = res[warmup:] if len(res) > warmup else res
        med = statistics.median(res)
        logger.info(f"median call time: {med} stdev: {statistics.stdev(res)} throughput: {1000.0/med:.1f} xfer/s/thread")
        
    





    


    




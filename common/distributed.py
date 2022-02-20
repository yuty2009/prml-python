
import os
import builtins
import torch
import torch.distributed as dist
    

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ and 'SLURM_NTASKS' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.world_size = int(os.environ['SLURM_NTASKS'])
        args.gpu = args.rank % torch.cuda.device_count()
    else: # for multiprocessing_distributed
        if args.rank == -1:
            args.rank = 0 # it is node rank for multiprocessing_distributed
        if 'WORLD_SIZE' in os.environ and args.world_size == -1:
            args.world_size = int(os.environ['WORLD_SIZE'])
        elif args.world_size == -1:
            args.world_size = 1 # it is number of nodes for multiprocessing_distributed

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.distributed = args.distributed and torch.cuda.is_available()

    args.ngpus = torch.cuda.device_count()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = args.ngpus * args.world_size

    return args


def init_distributed_process(args):
    # suppress printing if not master
    if args.distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.distributed:
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * args.ngpus + args.gpu
        else:
            args.dist_url = 'env://'
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

        if args.gpu is not None:
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / args.ngpus)
            args.workers = int((args.workers + args.ngpus - 1) / args.ngpus)

    if args.gpu is not None:
        args.device = torch.device('cuda', args.gpu)
        print("Use GPU: {} on Node: {} for training".format(args.gpu, args.rank // args.ngpus))

    print(args)
    return args


def convert_model(args, model):
    if args.distributed:
        # 
        # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
    else:
        args.dataparallel = True
        print("Use multi-GPU DataParallel for training")
        model = torch.nn.DataParallel(model).to(args.device)
    return model
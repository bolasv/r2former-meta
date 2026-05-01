import os
import torch
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Benchmarking Visual Geolocalization",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Training parameters
    parser.add_argument("--train_batch_size", type=int, default=4,help="Number of triplets (query, pos, negs) in a batch. Each triplet consists of 12 images")
    parser.add_argument("--infer_batch_size", type=int, default=16, help="Batch size for inference (caching and testing)")
    parser.add_argument("--rerank_batch_size", type=int, default=4, help="Batch size for inference (caching and testing)")
    parser.add_argument("--criterion", type=str, default='triplet', help='loss to be used',  choices=["triplet", "sare_ind", "sare_joint"])
    parser.add_argument("--margin", type=float, default=0.7, help="margin for the triplet loss")
    parser.add_argument("--epochs_num", type=int, default=50,help="number of epochs to train for")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.00001, help="_")
    parser.add_argument("--warmup", type=int, default=-1, help="_")
    parser.add_argument("--lr_crn_layer", type=float, default=5e-3, help="Learning rate for the CRN layer")
    parser.add_argument("--lr_crn_net", type=float, default=5e-4, help="Learning rate to finetune pretrained network when using CRN")
    parser.add_argument("--optim", type=str, default="adam", help="_", choices=["adam", "sgd","adamw"])
    parser.add_argument('--cos', action='store_true',help='use cosine lr schedule')
    parser.add_argument('--fix', type=int, default=1,help='use fixed global model')
    parser.add_argument('--freeze', type=int, default=0,help='freeze k layers of the model')
    parser.add_argument('--save_best', type=int, default=1,help='save the best R k')
    parser.add_argument('--finetune', type=int, default=0,help='enable the gradient flows back from reranking module to global retreival module')
    parser.add_argument('--test', action='store_true',help='use cosine lr schedule')
    parser.add_argument('--hypercolumn', type=int, default=0,help='use all layer for local feature')
    parser.add_argument('--reg_top', type=int, default=5,help='use all layer for local feature')
    parser.add_argument('--rerank_loss', type=str, default='ce',help='use triplet loss for rerank')
    parser.add_argument('--rerank_model', type=str, default='r2former',help='use triplet loss for rerank')
    parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int,help='learning rate schedule (when to drop lr by 10x)')
    parser.add_argument("--cache_refresh_rate", type=int, default=1000, help="How often to refresh cache, in number of queries")
    parser.add_argument("--queries_per_epoch", type=int, default=5000, help="How many queries to consider for one epoch. Must be multiple of cache_refresh_rate")
    parser.add_argument("--negs_num_per_query", type=int, default=10,help="How many negatives to consider per each query in the loss")
    parser.add_argument("--neg_samples_num", type=int, default=1000, help="How many negatives to use to compute the hardest ones")
    parser.add_argument("--neg_hardness", type=int, default=10, help="How many top negatives to be sampled from")
    parser.add_argument("--num_pairs", type=int, default=5, help="How many pairs to be sampled from")
    parser.add_argument("--local_dim", type=int, default=128,  help="local feature dimension")
    parser.add_argument("--num_local", type=int, default=196,  help="number of local features")
    parser.add_argument("--mining", type=str, default="partial", choices=["partial", "full", "random", "msls_weighted", 'global', 'global_combine'])
    # Model parameters
    parser.add_argument("--backbone", type=str, default="resnet18conv4",
                        choices=["alexnet", "vgg16", "resnet18conv4", "resnet18conv5", 
                                 "resnet50conv4", "resnet50conv5", "resnet101conv4", "resnet101conv5",
                                 "cct384", "vit", "deit", "deitBase", "resnet50"], help="_")
    parser.add_argument("--l2", type=str, default="before_pool", choices=["before_pool", "after_pool", "none"],
                        help="When (and if) to apply the l2 norm with shallow aggregation layers")
    parser.add_argument("--aggregation", type=str, default="netvlad", choices=["netvlad", "gem", "spoc", "mac", "rmac", "crn", "rrm",
                                                                               "cls", "seqpool", "none", "dynamic_gem"])
    parser.add_argument('--gem_mode', type=str, default='dynamic',
                        choices=['dynamic', 'fixed_multi', 'fixed_single'],
                        help='GeM mode for ablation (dynamic=proposed, fixed_multi=control, fixed_single=baseline)')
    # Meta-learning parameters (Paper Section 4.1)
    parser.add_argument('--meta_train', action='store_true', help='Enable meta-learning training mode')
    parser.add_argument('--gem_groups', type=int, default=8, help='GeM groups K (paper: 8)')
    parser.add_argument('--inner_steps', type=int, default=20, help='Inner loop steps tau (paper: 20)')
    parser.add_argument('--meta_train_steps', type=int, default=500, help='Meta-training steps (paper: 500)')
    parser.add_argument('--meta_batch_size', type=int, default=2, help='Meta batch size (paper: 2)')
    parser.add_argument('--outer_lr', type=float, default=1e-4, help='Outer loop LR (paper: 1e-4)')
    parser.add_argument('--use_minivelo', action='store_true', default=True, help='Use MiniVeLO optimizer')
    parser.add_argument('--no_minivelo', action='store_true', help='Disable MiniVeLO')
    parser.add_argument('--lstm_hidden_dim', type=int, default=128, help='LSTM hidden dim (paper: 128)')
    parser.add_argument('--gem_p_min', type=float, default=1.0, help='GeM power min (paper: 1.0)')
    parser.add_argument('--gem_p_max', type=float, default=10.0, help='GeM power max (paper: 10.0)')
    parser.add_argument('--paper_mode', action='store_true', default=True, help='5-dim stats (paper mode)')
    parser.add_argument('--convergence_baseline', type=float, default=79.0, help='Baseline mAP@1 for convergence')
    # Dynamic GeM parameters (for network.py compatibility)
    parser.add_argument('--dynamic_gem_init_p', type=float, default=3.0, help='Initial p for DynamicMultiGeM')
    parser.add_argument('--dynamic_gem_eps', type=float, default=1e-6, help='Epsilon for DynamicMultiGeM')
    parser.add_argument('--dynamic_gem_groups', type=int, default=8, help='Number of groups for DynamicMultiGeM')
    parser.add_argument('--dynamic_gem_hidden_dim', type=int, default=128, help='LSTM hidden dim for DynamicMultiGeM')
    parser.add_argument('--netvlad_clusters', type=int, default=64, help="Number of clusters for NetVLAD layer.")
    parser.add_argument('--pca_dim', type=int, default=None, help="PCA dimension (number of principal components). If None, PCA is not used.")
    parser.add_argument('--num_non_local', type=int, default=1, help="Num of non local blocks")
    parser.add_argument("--non_local", action='store_true', help="_")
    parser.add_argument('--channel_bottleneck', type=int, default=128, help="Channel bottleneck for Non-Local blocks")
    parser.add_argument('--fc_output_dim', type=int, default=384,
                        help="Output dimension of fully connected layer. If None, don't use a fully connected layer.")
    parser.add_argument('--pretrain', type=str, default="imagenet", choices=['imagenet', 'gldv2', 'places'],
                        help="Select the pretrained weights for the starting network")
    parser.add_argument("--off_the_shelf", type=str, default="imagenet", choices=["imagenet", "radenovic_sfm", "radenovic_gldv1", "naver"],
                        help="Off-the-shelf networks from popular GitHub repos. Only with ResNet-50/101 + GeM + FC 2048")
    parser.add_argument("--trunc_te", type=int, default=None, choices=list(range(0, 14)))
    parser.add_argument("--freeze_te", type=int, default=None, choices=list(range(-1, 14)))
    # Initialization parameters
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to load checkpoint from, for resuming training or testing.")
    # Other parameters
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num_workers", type=int, default=8, help="num_workers for all dataloaders")
    parser.add_argument('--resize', type=int, default=[224, 224], nargs=2, help="Resizing shape for images (HxW).")
    parser.add_argument('--test_method', type=str, default="hard_resize",
                        choices=["hard_resize", "single_query", "central_crop", "five_crops", "nearest_crop", "maj_voting"],
                        help="This includes pre/post-processing methods and prediction refinement")
    parser.add_argument("--majority_weight", type=float, default=0.01, 
                        help="only for majority voting, scale factor, the higher it is the more importance is given to agreement")
    parser.add_argument("--efficient_ram_testing", action='store_true', help="_")
    parser.add_argument("--val_positive_dist_threshold", type=int, default=25, help="_")
    parser.add_argument("--train_positives_dist_threshold", type=int, default=10, help="_")
    parser.add_argument('--recall_values', type=int, default=[1, 5, 10, 20, 100], nargs="+",
                        help="Recalls to be computed, such as R@5.")
    parser.add_argument('--features_dim', type=int, default=256)
    # Data augmentation parameters
    parser.add_argument("--brightness", type=float, default=None, help="_")
    parser.add_argument("--contrast", type=float, default=None, help="_")
    parser.add_argument("--saturation", type=float, default=None, help="_")
    parser.add_argument("--hue", type=float, default=None, help="_")
    parser.add_argument("--rand_perspective", type=float, default=None, help="_")
    parser.add_argument("--horizontal_flip", action='store_true', help="_")
    parser.add_argument("--random_resized_crop", type=float, default=None, help="_")
    parser.add_argument("--random_rotation", type=float, default=None, help="_")
    # Paths parameters
    parser.add_argument("--datasets_folder", type=str, default=None, help="Path with all datasets")
    parser.add_argument("--dataset_name", type=str, default="pitts30k", help="Relative path of the dataset")
    parser.add_argument("--pca_dataset_folder", type=str, default=None,
                        help="Path with images to be used to compute PCA (ie: pitts30k/images/train")
    parser.add_argument("--save_dir", type=str, default="default",
                        help="Folder name of the current run (saved in ./logs/)")
    args = parser.parse_args()
    
    if args.datasets_folder == None:
        try:
            args.datasets_folder = os.environ['DATASETS_FOLDER']
        except KeyError:
            raise Exception("You should set the parameter --datasets_folder or export " +
                            "the DATASETS_FOLDER environment variable as such \n" +
                            "export DATASETS_FOLDER=../datasets_vg/datasets")
    
    if args.aggregation == "crn" and args.resume == None:
        raise ValueError("CRN must be resumed from a trained NetVLAD checkpoint, but you set resume=None.")
    
    if args.queries_per_epoch % args.cache_refresh_rate != 0:
        raise ValueError("Ensure that queries_per_epoch is divisible by cache_refresh_rate, " +
                         f"because {args.queries_per_epoch} is not divisible by {args.cache_refresh_rate}")
    
    if torch.cuda.device_count() >= 2 and args.criterion in ['sare_joint', "sare_ind"]:
        raise NotImplementedError("SARE losses are not implemented for multiple GPUs, " +
                                  f"but you're using {torch.cuda.device_count()} GPUs and {args.criterion} loss.")
    
    if args.mining == "msls_weighted" and args.dataset_name != "msls":
        raise ValueError("msls_weighted mining can only be applied to msls dataset, but you're using it on {args.dataset_name}")
    
    if args.off_the_shelf in ["radenovic_sfm", "radenovic_gldv1", "naver"]:
        if args.backbone not in ["resnet50conv5", "resnet101conv5"] or args.aggregation != "gem" or args.fc_output_dim != 2048:
            raise ValueError("Off-the-shelf models are trained only with ResNet-50/101 + GeM + FC 2048")
    
    if args.pca_dim != None and args.pca_dataset_folder == None:
        raise ValueError("Please specify --pca_dataset_folder when using pca")
    
    # Set default attributes for meta-learning compatibility
    if not hasattr(args, 'work_with_tokens'):
        args.work_with_tokens = False
    if not hasattr(args, 'gem_mode'):
        args.gem_mode = 'dynamic'
    
    return args


import pickle

from utils.utils import *

ENCODER_DEFAULT_LR = {
    'default': 1e-3,
    'csqa': {
        'lstm': 3e-4,
        'openai-gpt': 1e-4,
        'bert-base-uncased': 3e-5,
        'bert-large-uncased': 2e-5,
        'roberta-large': 1e-5,
    },
    'obqa': {
        'lstm': 3e-4,
        'openai-gpt': 3e-5,
        'bert-base-cased': 1e-4,
        'bert-large-cased': 1e-4,
        'roberta-large': 1e-5,
    },
    'hfdata': {
        'lstm': 2e-5
    }
}

DATASET_LIST = ['csqa', 'obqa', 'socialiqa', 'hfdata']

DATASET_SETTING = {
    'csqa': 'inhouse',
    'obqa': 'official',
    'socialiqa': 'official',
}

DATASET_NO_TEST = ['socialiqa']

EMB_PATHS = {
    'transe': './data/transe/glove.transe.sgd.ent.npy',
    'lm': './data/transe/glove.transe.sgd.ent.npy',
    'numberbatch': './data/transe/concept.nb.npy',
    'tzw': './data/cpnet/tzw.ent.npy',
    'cui': './data/semmed/cui_embedding.npy'
}


def add_data_arguments(parser):
    parser.add_argument('--ent_emb', default=['cui'], choices=['transe', 'numberbatch', 'lm', 'tzw', 'cui'], nargs='+',
                        help='sources for entity embeddings')
    parser.add_argument('--ent_emb_paths', default=['./data/semmed/cui_embedding.npy'], nargs='+',
                        help='paths to entity embedding file(s)')
    parser.add_argument('--rel_emb_path', default='./data/transe/glove.transe.sgd.rel.npy',
                        help='paths to relation embedding file')
    parser.add_argument('-ds', '--dataset', default='hfdata', help='dataset name')
    parser.add_argument('-ih', '--inhouse', default=False, type=bool_flag, nargs='?', const=False,
                        help='run in-house setting')
    parser.add_argument('--inhouse_train_qids', default=None, help='qids of the in-house training set')
    parser.add_argument('--train_statements', default='./data/{dataset}/grounded/train_ground.jsonl')
    parser.add_argument('--dev_statements', default='./data/{dataset}/grounded/dev_ground.jsonl')
    parser.add_argument('--test_statements', default='./data/{dataset}/grounded/test_ground.jsonl')
    parser.add_argument('-sl', '--max_seq_len', default=50, type=int)
    parser.add_argument('--format', default=[],
                        choices=['add_qa_prefix', 'no_extra_sep', 'fairseq', 'add_prefix_space'], nargs='*')
    args, _ = parser.parse_known_args()
    parser.set_defaults(ent_emb_paths=[EMB_PATHS.get(s) for s in args.ent_emb],
                        inhouse=False,
                        inhouse_train_qids=None)
    data_splits = ('train', 'dev') if args.dataset in DATASET_NO_TEST else ('train', 'dev', 'test')
    for split in data_splits:
        for attribute in ('statements',):
            attr_name = f'{split}_{attribute}'
            parser.set_defaults(**{attr_name: getattr(args, attr_name).format(dataset=args.dataset)})
    if 'test' not in data_splits:
        parser.set_defaults(test_statements=None)


def add_encoder_arguments(parser):
    parser.add_argument('-enc', '--encoder', default='lstm', help='encoder type')
    parser.add_argument('--encoder_layer', default=-1, type=int,
                        help='encoder layer ID to use as features (used only by non-LSTM encoders)')
    parser.add_argument('-elr', '--encoder_lr', default=2e-5, type=float, help='learning rate')
    parser.add_argument('--n_diagnosis_codes', default=8692, type=int)
    parser.add_argument('--encoder_dim1', default=256, type=int, help='number of LSTM hidden units')
    parser.add_argument('--encoder_layer_num1', default=2, type=int, help='number of LSTM layers')
    parser.add_argument('--encoder_bidir1', default=True, type=bool_flag, nargs='?', const=True, help='use BiLSTM')
    parser.add_argument('--encoder_dropoute1', default=0.1, type=float, help='word dropout')
    parser.add_argument('--encoder_dropouti1', default=0.1, type=float, help='dropout applied to embeddings')
    parser.add_argument('--encoder_dropouth1', default=0.1, type=float, help='dropout applied to lstm hidden states')
    parser.add_argument('--encoder_pretrained_emb1', default='',
                        help='path to pretrained emb in .npy format')
    parser.add_argument('--encoder_freeze_emb1', default=False, type=bool_flag, nargs='?', const=False,
                        help='freeze lstm input embedding layer')
    parser.add_argument('--encoder_emb_size1', default=256, type=int, help='embedding size of input')
    parser.add_argument('--encoder_pooler1', default='max', choices=['max', 'mean'], help='pooling function')
    parser.add_argument('--lsan_emb_dim', default=128, type=int, help='embedding size of transformer')
    parser.add_argument('--att_heads', default=8, type=int, help='Number of Transformer Heads')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--layers', default=3, type=int, help='Number of Layers of Transformer')
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--retainEx_hidden_size', default=128, type=int)
    parser.add_argument('--retain_hidden_size', default=256, type=int)
    parser.add_argument('--retain_dropout_rate', default=0.5, type=float)
    parser.add_argument('--timeline_emb_dim', default=256, type=int)
    parser.add_argument('--timeline_hidden_dim', default=128, type=int)
    parser.add_argument('--timeline_att_dim', default=64, type=int)
    parser.add_argument('--timeline_dropout', default=0.5, type=float)
    parser.add_argument('--gruself_visit_size', default=256, type=int)
    parser.add_argument('--gruself_hidden_size', default=256, type=int)
    parser.add_argument('--gruself_dropout_rate', default=0.5, type=float)
    args, _ = parser.parse_known_args()
    parser.set_defaults(encoder_lr=ENCODER_DEFAULT_LR[args.dataset].get(args.encoder, ENCODER_DEFAULT_LR['default']))
    with open('./data/hfdata/hf_code2idx_new.pickle', 'rb') as fin:
        code2idx = pickle.load(fin)
        parser.set_defaults(n_diagnosis_codes=len(code2idx))


def add_optimization_arguments(parser):
    parser.add_argument('--loss', default='cross_entropy', choices=['margin_rank', 'BCE', 'cross_entropy'],
                        help='model type')
    parser.add_argument('--optim', default='radam', choices=['sgd', 'adam', 'adamw', 'radam'],
                        help='learning rate scheduler')
    parser.add_argument('--lr_schedule', default='fixed', choices=['fixed', 'warmup_linear', 'warmup_constant'],
                        help='learning rate scheduler')
    parser.add_argument('-bs', '--batch_size', default=64, type=int)
    parser.add_argument('--warmup_steps', type=float, default=150)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='max grad norm (0 to disable)')
    parser.add_argument('--weight_decay', default=1e-3, type=float, help='l2 weight decay strength')
    parser.add_argument('--n_epochs', default=200, type=int, help='total number of training epochs to perform.')
    parser.add_argument('-me', '--max_epochs_before_stop', default=15, type=int,
                        help='stop training if dev does not increase for N epochs')


def add_additional_arguments(parser):
    parser.add_argument('--log_interval', default=20, type=int)
    parser.add_argument('--cuda', default=True, type=bool_flag, nargs='?', const=True, help='use GPU')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--debug', default=False, type=bool_flag, nargs='?', const=True, help='run in debug mode')
    args, _ = parser.parse_known_args()
    if args.debug:
        parser.set_defaults(batch_size=1, log_interval=1, eval_interval=5)


def get_parser():
    """A helper function that handles the arguments that all models share"""
    parser = argparse.ArgumentParser(add_help=False)
    add_data_arguments(parser)
    add_encoder_arguments(parser)
    add_optimization_arguments(parser)
    add_additional_arguments(parser)
    return parser


def get_sand_config(args):
    lstm_config = {
        'vocab_size': args.n_diagnosis_codes,
    }
    return lstm_config


def get_gruself_config(args):
    lstm_config = {
        'n_diagnosis_codes': args.n_diagnosis_codes,
        'visit_size': args.gruself_visit_size,
        'hidden_size': args.gruself_hidden_size,
        'dropout_rate': args.gruself_dropout_rate
    }
    return lstm_config


def get_timeline_config(args):
    lstm_config = {
        'batch_size': args.batch_size,
        'embedding_dim': args.timeline_emb_dim,
        'hidden_dim': args.timeline_hidden_dim,
        'attention_dim': args.timeline_att_dim,
        'vocab_size': args.n_diagnosis_codes,
        'labelset_size': 2,
        'dropoutrate': args.timeline_dropout,
    }
    return lstm_config


def get_lstm_config_from_args(args):
    lstm_config = {
        'vocab_size': args.n_diagnosis_codes + 1,
        'emb_size': args.encoder_emb_size1,
        'hidden_size': args.encoder_dim1,
        'output_size': args.encoder_dim1,
        'num_layers': args.encoder_layer_num1,
        'bidirectional': args.encoder_bidir1,
        'emb_p': args.encoder_dropoute1,
        'input_p': args.encoder_dropouti1,
        'hidden_p': args.encoder_dropouth1,
        'pretrained_emb_or_path': args.encoder_pretrained_emb1,
        'freeze_emb': args.encoder_freeze_emb1,
        'pool_function': args.encoder_pooler1,
    }
    return lstm_config


def get_lsan_config_from_args(args):
    lstm_config = {
        'dict_len': args.n_diagnosis_codes,
        'embedding_dim': args.lsan_emb_dim,
        'transformer_hidden': args.lsan_emb_dim,
        'attn_heads': args.att_heads,
        'transformer_dropout': args.dropout,
        'transformer_layers': args.layers
    }
    return lstm_config


def get_hita_config_from_args(args):
    lstm_config = {
        'n_diagnosis_codes': args.n_diagnosis_codes,
        'batch_size': args.batch_size,
        'num_layers': args.num_layers,
        'dropout_rate': args.dropout_rate,
    }
    return lstm_config


def get_retainEx_config(args):
    lstm_config = {
        'input_size': args.n_diagnosis_codes,
        'hidden_size': args.retainEx_hidden_size,
        'num_classes': 2,
        'cuda_flag': args.cuda,
        'bidirectional': args.encoder_bidir1,
    }
    return lstm_config


def get_retain_config(args):
    lstm_config = {
        'n_diagnosis_codes': args.n_diagnosis_codes,
        'hidden_size': args.retain_hidden_size,
        'dropout_rate': args.retain_dropout_rate,
        'batch_size': args.batch_size,
    }
    return lstm_config

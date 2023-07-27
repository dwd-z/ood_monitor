from utils import log
import models.resnetv2 as resnetv2
import models.wrn as wrn

import torch
import torchvision as tv
import time

import numpy as np

from utils.test_utils import arg_parser, get_measures
import os

from sklearn.linear_model import LogisticRegressionCV
from torch.autograd import Variable
from utils.mahalanobis_lib import get_Mahalanobis_score


def make_id_ood(args, logger):
    """Returns train and validation datasets."""
    #TODO
    n_channels = 3
    test_tx = tv.transforms.Compose([
        tv.transforms.Resize((args.input_size, args.input_size)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(tuple(args.mean), tuple(args.std)),
    ])

    in_set = tv.datasets.ImageFolder(args.in_datadir, test_tx)
    out_set = tv.datasets.ImageFolder(args.out_datadir, test_tx)

    logger.info(f"Using an in-distribution set with {len(in_set)} images.")
    logger.info(f"Using an out-of-distribution set with {len(out_set)} images.")

    in_loader = torch.utils.data.DataLoader(
        in_set, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    out_loader = torch.utils.data.DataLoader(
        out_set, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    return in_set, out_set, in_loader, out_loader


def iterate_data_msp(data_loader, model):
    global device
    confs = []
    m = torch.nn.Softmax(dim=-1).to(device)
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.to(device)
            # compute output, measure accuracy and record loss.
            logits = model(x)

            conf, _ = torch.max(m(logits), dim=-1)
            confs.extend(conf.data.cpu().numpy())
    return np.array(confs)


def iterate_data_odin(data_loader, model, epsilon, temper, logger):
    criterion = torch.nn.CrossEntropyLoss().to(device)
    confs = []
    for b, (x, y) in enumerate(data_loader):
        x = Variable(x.to(device), requires_grad=True)
        outputs = model(x)

        maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)
        outputs = outputs / temper

        labels = Variable(torch.LongTensor(maxIndexTemp).to(device))
        loss = criterion(outputs, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(x.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        # TODO
        gradient[:,0] = (gradient[:,0] )/(63.0/255.0)
        gradient[:,1] = (gradient[:,1] )/(62.1/255.0)
        gradient[:,2] = (gradient[:,2] )/(66.7/255.0)

        # Adding small perturbations to images
        tempInputs = torch.add(x.data, -epsilon, gradient)
        outputs = model(Variable(tempInputs))
        outputs = outputs / temper
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)

        confs.extend(np.max(nnOutputs, axis=1))
        if b % 100 == 0:
            logger.info('{} batches processed'.format(b))

        # debug
        # if b > 500:
        #    break

    return np.array(confs)


def iterate_data_energy(data_loader, model, temper):
    confs = []
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.to(device)
            # compute output, measure accuracy and record loss.
            logits = model(x)

            conf = temper * torch.logsumexp(logits / temper, dim=1)
            confs.extend(conf.data.cpu().numpy())
    return np.array(confs)


def iterate_data_mahalanobis(data_loader, model, num_classes, sample_mean, precision,
                             num_output, magnitude, regressor, logger):
    confs = []
    for b, (x, y) in enumerate(data_loader):
        if b % 10 == 0:
            logger.info('{} batches processed'.format(b))
        x = x.to(device)

        Mahalanobis_scores = get_Mahalanobis_score(x, model, num_classes, sample_mean, precision, num_output, magnitude)
        scores = -regressor.predict_proba(Mahalanobis_scores)[:, 1]
        confs.extend(scores)
    return np.array(confs)


def iterate_data_gradnorm(data_loader, model, model_name, temperature, num_classes):
    confs = []
    logsoftmax = torch.nn.LogSoftmax(dim=-1).to(device)
    for b, (x, y) in enumerate(data_loader):
        if b % 100 == 0:
            print('{} batches processed'.format(b))
        inputs = Variable(x.to(device), requires_grad=True)

        model.zero_grad()
        outputs = model(inputs)
        targets = torch.ones((inputs.shape[0], num_classes)).to(device)
        outputs = outputs / temperature
        loss = torch.mean(torch.sum(-targets * logsoftmax(outputs), dim=-1))

        loss.backward()
        
        if model_name == 'resnetv2':
            layer_grad = model.head.conv.weight.grad.data
        elif model_name == 'wrn':
            layer_grad = model.fc.weight.grad.data

        layer_grad_norm = torch.sum(torch.abs(layer_grad)).cpu().numpy()
        confs.append(layer_grad_norm)

    return np.array(confs)


def run_eval(model, in_loader, out_loader, logger, args, num_classes):
    # switch to evaluate mode
    model.eval()

    logger.info("Running test...")
    logger.flush()

    if args.score == 'MSP':
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_msp(in_loader, model)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_msp(out_loader, model)
    elif args.score == 'ODIN':
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_odin(in_loader, model, args.epsilon_odin, args.temperature_odin, logger)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_odin(out_loader, model, args.epsilon_odin, args.temperature_odin, logger)
    elif args.score == 'Energy':
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_energy(in_loader, model, args.temperature_energy)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_energy(out_loader, model, args.temperature_energy)
    elif args.score == 'Mahalanobis':
        sample_mean, precision, lr_weights, lr_bias, magnitude = np.load(
            os.path.join(args.mahalanobis_param_path, 'results.npy'), allow_pickle=True)
        sample_mean = [s.to(device) for s in sample_mean]
        precision = [p.to(device) for p in precision]

        regressor = LogisticRegressionCV(cv=2).fit([[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]],
                                                   [0, 0, 1, 1])

        regressor.coef_ = lr_weights
        regressor.intercept_ = lr_bias

        temp_x = torch.rand(2, 3, 480, 480)
        temp_x = Variable(temp_x).to(device)
        temp_list = model(x=temp_x, layer_index='all')[1]
        num_output = len(temp_list)

        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_mahalanobis(in_loader, model, num_classes, sample_mean, precision,
                                             num_output, magnitude, regressor, logger)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_mahalanobis(out_loader, model, num_classes, sample_mean, precision,
                                              num_output, magnitude, regressor, logger)
    elif args.score == 'GradNorm':
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_gradnorm(in_loader, model, args.model_name, args.temperature_gradnorm, num_classes)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_gradnorm(out_loader, model, args.model_name, args.temperature_gradnorm, num_classes)
    else:
        raise ValueError("Unknown score type {}".format(args.score))

    if (args.num_ood) is not None:
        out_scores = out_scores[:(args.num_ood)]
    in_examples = in_scores.reshape((-1, 1))
    out_examples = out_scores.reshape((-1, 1))

    auroc, aupr_in, aupr_out, fpr95 = get_measures(in_examples, out_examples)

    logger.info('============Results for {}============'.format(args.score))
    logger.info('AUROC: {}'.format(auroc))
    logger.info('AUPR (In): {}'.format(aupr_in))
    logger.info('AUPR (Out): {}'.format(aupr_out))
    logger.info('FPR95: {}'.format(fpr95))

    logger.flush()


def main(args):
    global device
    logger = log.setup_logger(args)

    torch.backends.cudnn.benchmark = True

    if args.score == 'GradNorm':
        args.batch = 1

    # Load ID and OoD datasets
    in_set, out_set, in_loader, out_loader = make_id_ood(args, logger)

    # Build model
    logger.info(f"Loading model from {args.model_path}")

    state_dict = torch.load(args.model_path)
    if args.model_name == 'resnetv2':
        model = resnetv2.KNOWN_MODELS[args.model](head_size=len(in_set.classes))
        model.load_state_dict_custom(state_dict['model'])
    elif args.model_name == 'wrn':
        model = wrn.KNOWN_MODELS[args.model](head_size=len(in_set.classes), dropRate=args.dropRate)
        model.load_state_dict(state_dict)

    if args.score != 'GradNorm':
        model = torch.nn.DataParallel(model)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    # OoD detection performance evaluation
    start_time = time.time()
    run_eval(model, in_loader, out_loader, logger, args, num_classes=len(in_set.classes))
    end_time = time.time()

    logger.info("Total running time: {}".format(end_time - start_time))


if __name__ == "__main__":
    parser = arg_parser()

    parser.add_argument("--in_datadir", help="Path to the in-distribution data folder.")
    parser.add_argument("--out_datadir", help="Path to the out-of-distribution data folder.")
    parser.add_argument("--num_ood", default=None, type=int, help="Number of OoD samples taken for the test.")

    parser.add_argument('--score', choices=['MSP', 'ODIN', 'Energy', 'Mahalanobis', 'GradNorm'], default='GradNorm')

    # arguments for ODIN
    parser.add_argument('--temperature_odin', default=1000, type=int,
                        help='temperature scaling for odin')
    parser.add_argument('--epsilon_odin', default=0.0, type=float,
                        help='perturbation magnitude for odin')

    # arguments for Energy
    parser.add_argument('--temperature_energy', default=1, type=int,
                        help='temperature scaling for energy')

    # arguments for Mahalanobis
    parser.add_argument('--mahalanobis_param_path', default='checkpoints/finetune/tune_mahalanobis',
                        help='path to tuned mahalanobis parameters')

    # arguments for GradNorm
    parser.add_argument('--temperature_gradnorm', default=1, type=int,
                        help='temperature scaling for GradNorm')

    main(parser.parse_args())

import csv
from argparse import ArgumentParser
import numpy as np
import random
import torch
from torch.optim import Adam
from torch.optim import lr_scheduler
import tensorflow as tf
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import pmiqd

from tensorboardX import SummaryWriter




# gpus = tf.config.experimental.list_physical_devices('GPU')
# print("Num GPUs:", len(gpus))
# cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
# print(gpus, cpus)



def run(args):

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    # print("Num GPUs:", len(gpus))
    # cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
    # print(gpus, cpus)
    # print(torch.cuda.is_available())
    device = torch.device('cuda')
    # print(device)
    train_loader, val_loader, test_loader, scale = pmiqd.get_data_loaders(args)
    lr_ratio = 1
    model = pmiqd.model()
    model_dict = model.state_dict()
    model.load_state_dict(model_dict)
    thelog = SummaryWriter(log_dir=args.log_dir)
    model = model.to(device)
    all_params = model.parameters()
    re_params = []
    for name, p in model.named_parameters():
        if name.find('fc') >= 0:
            re_params.append(p)
    re_id = list(map(id, re_params))
    feature = list(filter(lambda p: id(p) not in re_id, all_params))
    optimizer = Adam([{'params': re_params},
                      {'params':  feature, 'lr': args.lr * lr_ratio}],
                     lr=args.lr, weight_decay = 0)
    schestep = lr_scheduler.StepLR(optimizer, step_size=100, gamma=args.decay_ratio)
    global criterion
    criterion = -1
    trainer = create_supervised_trainer(model, optimizer, pmiqd.LossFuc(), device=device)
    evaluator = create_supervised_evaluator(model,metrics={'Performance': pmiqd.PerformanceEva()},device=device)
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        thelog.add_scalar("training/loss", scale * engine.state.output, engine.state.iteration)
    # fvalout = open('Validoutq.csv', 'w', newline='' "")
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        SROCC, KROCC, PLCC, RMSE, MAE, OR ,sq,q= metrics['IQA_performance']
        print("Validation Results - Epoch: {} SROCC: {:.4f} KROCC: {:.4f} PLCC: {:.4f} RMSE: {:.4f} MAE: {:.4f} OR: {:.2f}%"
              .format(engine.state.epoch, SROCC, KROCC, PLCC, scale * RMSE, scale * MAE, 100 * OR))
        thelog.add_scalar("SROCC/validation", SROCC, engine.state.epoch)
        thelog.add_scalar("KROCC/validation", KROCC, engine.state.epoch)
        thelog.add_scalar("PLCC/validation", PLCC, engine.state.epoch)
        thelog.add_scalar("RMSE/validation", scale * RMSE, engine.state.epoch)
        thelog.add_scalar("MAE/validation", scale * MAE, engine.state.epoch)
        thelog.add_scalar("OR/validation", OR, engine.state.epoch)
        q = [SROCC,PLCC,RMSE,KROCC,MAE,OR]
        # csv_writer = csv.writer(fvalout)
        # csv_writer.writerow(q)
        np.savetxt('validation_matrix.csv', q, delimiter=',')
        schestep.step(engine.state.epoch)
        global criterion
        global best_epoch
        if SROCC > criterion and engine.state.epoch/args.epochs > 1/6:  #
            criterion = SROCC
            best_epoch = engine.state.epoch
            try:
                torch.save(model.module.state_dict(), args.trained_model_file)
            except:
                torch.save(model.state_dict(), args.trained_model_file)
    # ft = open('TESToutq.csv', 'w', newline='' "")
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_testing_results(engine):
        # if args.test_during_training:
            evaluator.run(test_loader)
            thedata = evaluator.state.metrics
            SROCC, KROCC, PLCC, RMSE, MAE, OR ,sq,q = thedata['IQA_performance']
            print("Testing Results    - Epoch: {} SROCC: {:.4f} KROCC: {:.4f} PLCC: {:.4f} RMSE: {:.4f} MAE: {:.4f} OR: {:.2f}%"
                  .format(engine.state.epoch, SROCC, KROCC, PLCC, scale * RMSE, scale * MAE, 100 * OR))
            thelog.add_scalar("SROCC/testing", SROCC, engine.state.epoch)
            thelog.add_scalar("KROCC/testing", KROCC, engine.state.epoch)
            thelog.add_scalar("PLCC/testing", PLCC, engine.state.epoch)
            thelog.add_scalar("RMSE/testing", scale * RMSE, engine.state.epoch)
            thelog.add_scalar("MAE/testing", scale * MAE, engine.state.epoch)
            thelog.add_scalar("OR/testing", OR, engine.state.epoch)
            mat = [SROCC,PLCC,RMSE,KROCC,MAE,OR]
            # csv_writer = csv.writer(ft)
            # csv_writer.writerow(mat)
            schestep.step(engine.state.epoch)
    @trainer.on(Events.COMPLETED)
    def final_testing_results(engine):
        global best_epoch
        model.load_state_dict(torch.load(args.trained_model_file))
        evaluator.run(test_loader)
        metrics = evaluator.state.metrics
        SROCC, KROCC, PLCC, RMSE, MAE, OR ,sq,q= metrics['IQA_performance']
        print("Final Test Results - Epoch: {} SROCC: {:.4f} KROCC: {:.4f} PLCC: {:.4f} RMSE: {:.4f} MAE: {:.4f} OR: {:.2f}%"
            .format(best_epoch, SROCC, KROCC, PLCC, scale * RMSE, scale * MAE, 100 * OR))
    trainer.run(train_loader, max_epochs=args.epochs)
    thelog.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--database', default='PMIQD', type=str, help='database name')
    parser.add_argument('--patch_size', default=256)
    parser.add_argument('--n_patches', default=32)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=20000)
    parser.add_argument('--decay_ratio', type=int, default=0.7)
    args = parser.parse_args()
    args.data_info = './data/PMIQD.mat'
    args.im_dir = './data/dataset/PMIQD'
    args.trained_model_file = 'modelfile/dataset={}-learningrate={}-batchsize={}'.format(args.database, args.lr,args.batch_size)
    args.log_dir = 'logs/dataset={}-learningrate={}-batchsize={}'.format(args.database, args.lr,args.batch_size)
    torch.manual_seed(args.seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)
    run(args)

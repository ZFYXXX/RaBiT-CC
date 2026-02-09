from utils.evaluation import eval_game, eval_relative
from utils.trainer import Trainer
from utils.helper import Save_Handle, AverageMeter
import os
import sys
import time
import torch
import torch.nn.functional as F

from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.tensorboard import SummaryWriter
import logging
import numpy as np
import atexit
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from datasets.crowd_drone import Crowd_Drone
from datasets.crowd_rgbtcc import Crowd_RGBTCC
from losses.bay_loss import Bay_Loss
from losses.post_prob import Post_Prob
from losses.bpl import BPL_Loss
from models.RaBiT_Model import fusion_model


def train_collate(batch):
    transposed_batch = list(zip(*batch))
    if type(transposed_batch[0][0]) == list:
        rgb_list = [item[0] for item in transposed_batch[0]]
        t_list = [item[1] for item in transposed_batch[0]]
        rgb = torch.stack(rgb_list, 0)
        t = torch.stack(t_list, 0)
        images = [rgb, t]
    else:
        images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
    st_sizes = torch.FloatTensor(transposed_batch[2])
    return images, points, st_sizes


class RegTrainer(Trainer):
    def setup(self):
        """initial the datasets, model, loss and optimizer"""
        args = self.args

        log_dir = os.path.join(self.save_dir, 'tensorboard_logs')
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        atexit.register(self.writer.close)  
        logging.info(f"TensorBoard logs will be saved to: {log_dir}")
        
        if torch.cuda.is_available():

            self.device_count = torch.cuda.device_count()
            logging.info("available gpu count: {}".format(self.device_count))
            self.device = torch.device("cuda:{}".format(args.device))
            logging.info('using gpu {}'.format(args.device))
                
        else:
            raise Exception("gpu is not available")

        self.downsample_ratio = args.downsample_ratio
        self.datasets = None
        self.dataloaders = None
        if args.dataset == 'RGBTCC':
            self.datasets = {x: Crowd_RGBTCC(os.path.join(args.data_dir, x),
                                    args.crop_size,
                                    args.downsample_ratio,
                                    x) for x in ['train', 'val', 'test']}
            self.dataloaders = {x: DataLoader(self.datasets[x],
                                            collate_fn=(train_collate
                                                        if x == 'train' else default_collate),
                                            batch_size=(args.batch_size
                                            if x == 'train' else 1),
                                            shuffle=(True if x == 'train' else False),
                                            num_workers=args.num_workers*self.device_count,
                                            pin_memory=(True if x == 'train' else False))
                                for x in ['train', 'val', 'test']}
        elif args.dataset == 'DroneRGBT':
            self.datasets = {x: Crowd_Drone(os.path.join(args.data_dir, x),
                                    args.crop_size,
                                    args.downsample_ratio,
                                    x) for x in ['train', 'test']}
            self.dataloaders = {x: DataLoader(self.datasets[x],
                                            collate_fn=(train_collate
                                                        if x == 'train' else default_collate),
                                            batch_size=(args.batch_size
                                            if x == 'train' else 1),
                                            shuffle=(True if x == 'train' else False),
                                            num_workers=args.num_workers*self.device_count,
                                            pin_memory=(True if x == 'train' else False))
                                for x in ['train', 'test']}
        else:
            raise Exception("The dataset is not implemented!")

        self.model = fusion_model(
            construct_method=args.constr_hg,
            k_interact=args.constr_k,
            thresh_interact=args.constr_threshold
        )

        logging.info('model params (M): {}'.format(sum(p.numel() for p in self.model.parameters()) / 1e6))
        
        self.model.to(self.device)
        
        params = []
        for key, value in dict(self.model.named_parameters()).items():
            if 'pvt_backbone_rgb' in key or 'pvt_backbone_t' in key:
                params += [{'params': value, 'lr': args.lr * 0.1}]
            else:
                params += [{'params': value}]
        self.optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
        
        #  ******** You can try to use scheduler or not. **********
        # self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[400, 800, 1200], gamma=0.1)        
        

        self.start_epoch = 0
        if args.resume:
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume, self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suf == 'pth':
                self.model.load_state_dict(torch.load(args.resume, self.device))

        self.post_prob = Post_Prob(args.sigma,
                                   args.crop_size,
                                   args.downsample_ratio,
                                   args.background_ratio,
                                   args.use_background,
                                   self.device)
        self.criterion = Bay_Loss(args.use_background, self.device)
        
        self.bpl_criterion = BPL_Loss(
            window_size=8, 
            stride=8,      
            delta=0.5,      
            downsample_ratio=args.downsample_ratio 
        )

        
        self.lambda_aux_bayes = 0.5  
        self.lambda_bpl = 0.1        

        self.save_list = Save_Handle(max_num=args.max_model_num)
        

        self.best_game0 = np.inf
        self.best_game3 = np.inf
        self.best_count = 0
        self.best_count_1 = 0
        
        self.best_test_game0 = np.inf
        self.best_test_game1 = np.inf
        self.best_test_game2 = np.inf
        self.best_test_game3 = np.inf
        self.best_test_mse = np.inf
        self.best_test_count = 0
        self.best_test_count_1 = 0

    def train(self):
        """training process"""
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch):

            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
            self.epoch = epoch
            self.train_eopch()
            
            if args.dataset!='DroneRGBT': 
                if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                    self.val_epoch()

            if epoch % args.test_epoch == 0 and epoch >= args.test_start:
                self.test_epoch()


    def train_eopch(self):
        epoch_loss = AverageMeter()
        epoch_game = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_start = time.time()
        self.model.train()  

        for step, (inputs, points, st_sizes) in enumerate(self.dataloaders['train']):

            if type(inputs) == list:
                inputs[0] = inputs[0].to(self.device)
                inputs[1] = inputs[1].to(self.device)
            else:
                inputs = inputs.to(self.device)
            st_sizes = st_sizes.to(self.device)
            gd_count = np.array([len(p) for p in points], dtype=np.float32)
            points = [p.to(self.device) for p in points]

            with torch.set_grad_enabled(True):

                
                D_F, D_R, D_T, reli_maps_R, reli_maps_T = self.model(inputs)

                prob_list = self.post_prob(points, st_sizes)

                loss_bayes_F = self.criterion(prob_list, D_F)

                loss_bayes_R = self.criterion(prob_list, D_R)
                loss_bayes_T = self.criterion(prob_list, D_T)

                r_R_bpl = F.adaptive_avg_pool2d(reli_maps_R[0], output_size=D_R.shape[2:])
                r_T_bpl = F.adaptive_avg_pool2d(reli_maps_T[0], output_size=D_T.shape[2:])

                loss_bpl = self.bpl_criterion(D_R, D_T, r_R_bpl, r_T_bpl, points)

                total_loss = loss_bayes_F + \
                            self.lambda_aux_bayes * (loss_bayes_R + loss_bayes_T) + \
                            self.lambda_bpl * loss_bpl

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                if type(inputs) == list:
                    N = inputs[0].size(0)
                else:
                    N = inputs.size(0)

                pre_count = torch.sum(D_F.view(N, -1), dim=1).detach().cpu().numpy()

                res = pre_count - gd_count

                epoch_loss.update(total_loss.item(), N) 
                epoch_mse.update(np.mean(res * res), N)
                epoch_game.update(np.mean(abs(res)), N)

                global_step = self.epoch * len(self.dataloaders['train']) + step
                self.writer.add_scalar('Train_Step/Loss_Total', total_loss.item(), global_step)
                self.writer.add_scalar('Train_Step/Loss_Bayes_F', loss_bayes_F.item(), global_step)
                self.writer.add_scalar('Train_Step/Loss_Bayes_R', loss_bayes_R.item(), global_step)
                self.writer.add_scalar('Train_Step/Loss_Bayes_T', loss_bayes_T.item(), global_step)
                self.writer.add_scalar('Train_Step/Loss_BPL', loss_bpl.item(), global_step)
                
        
        logging.info('Epoch {} Train, Loss: {:.2f}, GAME0: {:.2f} MSE: {:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, epoch_loss.get_avg(), epoch_game.get_avg(), np.sqrt(epoch_mse.get_avg()),
                             time.time()-epoch_start))

        avg_train_loss = epoch_loss.get_avg()
        avg_train_game0 = epoch_game.get_avg()
        avg_train_rmse = np.sqrt(epoch_mse.get_avg())
        
        self.writer.add_scalar('Train_Epoch/Loss', avg_train_loss, self.epoch)
        self.writer.add_scalar('Train_Epoch/GAME0', avg_train_game0, self.epoch)
        self.writer.add_scalar('Train_Epoch/RMSE', avg_train_rmse, self.epoch)
        
        
        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('Train_Epoch/LearningRate', current_lr, self.epoch)

        model_state_dic = self.model.state_dict()
        save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
        torch.save({
            'epoch': self.epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_state_dict': model_state_dic
        }, save_path)
        self.save_list.append(save_path)  

    def val_epoch(self):
        args = self.args
        self.model.eval() 

        game = [0, 0, 0, 0]
        mse = [0, 0, 0, 0]
        total_relative_error = 0

        for inputs, target, name in self.dataloaders['val']:
            if type(inputs) == list:
                inputs[0] = inputs[0].to(self.device)
                inputs[1] = inputs[1].to(self.device)
            else:
                inputs = inputs.to(self.device)

            if type(inputs) == list:
                assert inputs[0].size(0) == 1
            else:
                assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                for L in range(4):
                    abs_error, square_error = eval_game(outputs, target, L)
                    game[L] += abs_error
                    mse[L] += square_error
                relative_error = eval_relative(outputs, target)
                total_relative_error += relative_error

        N = len(self.dataloaders['val'])
        game = [m / N for m in game]
        mse = [torch.sqrt(m / N) for m in mse]
        total_relative_error = total_relative_error / N

        logging.info('Epoch {} Val{}, '
                     'GAME0 {game0:.2f} GAME1 {game1:.2f} GAME2 {game2:.2f} GAME3 {game3:.2f} MSE {mse:.2f} Re {relative:.4f}, '
                     .format(self.epoch, N, game0=game[0], game1=game[1], game2=game[2], game3=game[3], mse=mse[0], relative=total_relative_error
                             )
                     )

        self.writer.add_scalar('Val/GAME0', game[0], self.epoch)
        self.writer.add_scalar('Val/GAME1', game[1], self.epoch)
        self.writer.add_scalar('Val/GAME2', game[2], self.epoch)
        self.writer.add_scalar('Val/GAME3', game[3], self.epoch)
        self.writer.add_scalar('Val/RMSE', mse[0], self.epoch)
        self.writer.add_scalar('Val/RelativeError', total_relative_error, self.epoch)
    def test_epoch(self):
        self.model.eval() 

        game = [0, 0, 0, 0]
        mse = [0, 0, 0, 0]
        total_relative_error = 0

        for inputs, target, name in self.dataloaders['test']:
            if type(inputs) == list:
                inputs[0] = inputs[0].to(self.device)
                inputs[1] = inputs[1].to(self.device)
            else:
                inputs = inputs.to(self.device)

            if type(inputs) == list:
                assert inputs[0].size(0) == 1
            else:
                assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                for L in range(4):
                    abs_error, square_error = eval_game(outputs, target, L)
                    game[L] += abs_error
                    mse[L] += square_error
                relative_error = eval_relative(outputs, target)
                total_relative_error += relative_error

        N = len(self.dataloaders['test'])
        game = [m / N for m in game]
        mse = [torch.sqrt(m / N) for m in mse]
        total_relative_error = total_relative_error / N

        logging.info('Epoch {} Test{}, '
                     'GAME0 {game0:.2f} GAME1 {game1:.2f} GAME2 {game2:.2f} GAME3 {game3:.2f} MSE {mse:.2f} Re {relative:.4f}, '
                     .format(self.epoch, N, game0=game[0], game1=game[1], game2=game[2], game3=game[3], mse=mse[0],
                             relative=total_relative_error
                             )
                     )
        
        self.writer.add_scalar('Test/GAME0', game[0], self.epoch)
        self.writer.add_scalar('Test/GAME1', game[1], self.epoch)
        self.writer.add_scalar('Test/GAME2', game[2], self.epoch)
        self.writer.add_scalar('Test/GAME3', game[3], self.epoch)
        self.writer.add_scalar('Test/RMSE', mse[0], self.epoch)
        self.writer.add_scalar('Test/RelativeError', total_relative_error, self.epoch)

        model_state_dic = self.model.state_dict()
        
        if bool(game[0] < self.best_test_game0):
            self.best_test_game0 = game[0]
            logging.info("*** Best Test GAME0 {:.3f} model epoch {}".format(self.best_test_game0, self.epoch))
            torch.save(model_state_dic, os.path.join(self.save_dir, 'best_test_model_game0.pth'))
            
        if bool(game[1] < self.best_test_game1):
            self.best_test_game1 = game[1]
            logging.info("*** Best Test GAME1 {:.3f} model epoch {}".format(self.best_test_game1, self.epoch))
            torch.save(model_state_dic, os.path.join(self.save_dir, 'best_test_model_game1.pth'))
            
        if bool(game[2] < self.best_test_game2):
            self.best_test_game2 = game[2]
            logging.info("*** Best Test GAME2 {:.3f} model epoch {}".format(self.best_test_game2, self.epoch))
            torch.save(model_state_dic, os.path.join(self.save_dir, 'best_test_model_game2.pth'))
            
        if bool(game[3] < self.best_test_game3):
            
            self.best_test_game3 = game[3]
            logging.info("*** Best Test GAME3 {:.3f} model epoch {}".format(self.best_test_game3, self.epoch))
            torch.save(model_state_dic, os.path.join(self.save_dir, 'best_test_model_game3.pth'))
            
        if bool(mse[0] < self.best_test_mse):
            self.best_test_mse = mse[0]
            logging.info("*** Best Test MSE {:.3f} model epoch {}".format(self.best_test_mse, self.epoch))
            torch.save(model_state_dic, os.path.join(self.save_dir, 'best_test_model_mse.pth'))
            
            
    




import time
from datetime import datetime
import sys
import random
import argparse
from net import *
import torch.optim as optim
from utils import *
from data.datasets import get_loader, get_test_loader
from config import config

def train_one_epoch(epoch, net, train_loader, optimizer_G, device, logger):
    global global_step
    net.train()
    elapsed, losses, psnrs, psnr_jsccs_1,psnr_jsccs_2, bpps, cbrs, kls = [AverageMeter() for _ in range(8)]
    metrics = [elapsed, losses, psnrs, psnr_jsccs_1,psnr_jsccs_2, bpps, cbrs, kls]
    for batch_idx, input_image in enumerate(train_loader):
        optimizer_G.zero_grad()
        start_time = time.time()
        input_image = input_image.cuda()
        global_step += 1
        loss,psnr_ntc,psnr_plit_1,psnr_plit_2,kl,cbr_z,bpp = net(input_image)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        optimizer_G.step()
        psnr = psnr_ntc

        cbrs.update(cbr_z)
        kls.update(kl.item())
        elapsed.update(time.time() - start_time)
        losses.update(loss.item())
        bpps.update(bpp.item())
        psnr_jsccs_1.update(psnr_plit_1)
        psnr_jsccs_2.update(psnr_plit_2)
        psnrs.update(psnr)

        if (global_step % config.print_step) == 0 or global_step==1:
            process = (global_step % train_loader.__len__()) / (train_loader.__len__()) * 100.0
            log = (' | '.join([
                f'Step [{global_step % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]',
                f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                f'Time {elapsed.avg:.2f}',
                f'PSNR_JSCC_1 {psnr_jsccs_1.val:.2f} ({psnr_jsccs_1.avg:.2f})',
                f'PSNR_JSCC_2 {psnr_jsccs_2.val:.2f} ({psnr_jsccs_2.avg:.2f})',
                f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                f'PSNR_NTC {psnrs.val:.2f} ({psnrs.avg:.2f})',
                f'Bpp {bpps.val:.2f} ({bpps.avg:.2f})',
                f'Kl {kls.val:.2f} ({kls.avg:.2f})',
                f'Epoch {epoch}',
            ]))
            logger.info(log)
            for i in metrics:
                i.clear()


def test(net, test_loader, logger):
    with torch.no_grad():
        net.eval()
        elapsed, losses, psnrs, psnr_jsccs_1, psnr_jsccs_2, bpps, cbrs, kls = [AverageMeter() for _ in range(8)]
        PSNR_list = []
        CBR_list = []
        for batch_idx, input_image in enumerate(test_loader):
            input_image = input_image.cuda()
            start_time = time.time()
            loss,psnr_ntc,psnr_plit_1,psnr_plit_2,kl,cbr_z,bpp = net(input_image,lmb=128)
            psnr = psnr_ntc
            cbrs.update(cbr_z)
            kls.update(kl.item())
            elapsed.update(time.time() - start_time)
            losses.update(loss.item())
            bpps.update(bpp.item())
            psnr_jsccs_1.update(psnr_plit_1)
            psnr_jsccs_2.update(psnr_plit_2)
            psnrs.update(psnr)

            log = (' | '.join([
                f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                f'Time {elapsed.val:.2f}',
                f'PSNR_JSCC_1 {psnr_jsccs_1.val:.2f} ({psnr_jsccs_1.avg:.2f})',
                f'PSNR_JSCC_2 {psnr_jsccs_2.val:.2f} ({psnr_jsccs_2.avg:.2f})',
                f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                f'PSNR_NTC {psnrs.val:.2f} ({psnrs.avg:.2f})',
                f'Bpp {bpps.val:.2f} ({bpps.avg:.2f})',
                f'Kl {kls.val:.4f} ({kls.avg:.4f})',
            ]))
            logger.info(log)
            PSNR_list.append(psnr_plit_2)
            CBR_list.append(cbr_z)

    # Here, the channel bandwidth cost of side info \bar{k} is transmitted by a capacity-achieving channel code. Note
    # that, the side info should be transmitted through entropy coding and channel coding, which will be addressed in
    # future releases.

    # capacity-achieving channel code
    first_stage=np.log2(16) * 1 / (64 * 64 * 3)/1
    second_stage=np.log2(16) * 2/ (32 * 32 * 3)/4
    third_stage=np.log2(32) * 3 / (16 * 16 * 3)/4
    fourth_stage=np.log2(8) * 3 /(8 * 8 * 3)/16
    cbr_sideinfo = (first_stage+second_stage+third_stage+fourth_stage)/np.log2(1 + 10 ** (config.channel['chan_param'] / 10))

    # 2/3 rate LDPC + 16QAM for AWGN SNR=10dB
    # cbr_sideinfo = np.log2(config.multiple_rate.__len__()) / (16 * 16 * 8)
    logger.info(f'Finish test! Average PSNR={psnr_jsccs_2.avg:.4f}dB,CBR={cbrs.avg:.4f}, CBR_all={cbrs.avg + cbr_sideinfo:.4f}')
    return losses.avg



def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training/testing script.")
    parser.add_argument(
        "-p",
        "--phase",
        default='test',  # train
        type=str,
        help="Train or Test",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=5000,
        type=int,
        help="Number of epochs (default: %(default)s)"
    )
    parser.add_argument("--cuda", default=True, action="store_true", help="Use cuda")
    parser.add_argument(
        "--gpu-id",
        type=str,
        default=2,
        help="GPU ids (default: %(default)s)",
    )
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=float, default=1024, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        '--name',
        default=datetime.now().strftime('%Y-%m-%d_%H_%M_%S'),
        type=str,
        help='Result dir name',
    )
    parser.add_argument(
        '--save_log', action='store_true', default=True, help='Save log to disk'
    )
    parser.add_argument("--checkpoint",
                        default='PLIT_2_phases_quality_1',
                        type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    config.device = device

    workdir, logger = logger_configuration(args.name, phase=args.phase, save_log=args.save_log)
    config.logger = logger
    logger.info(config.__dict__)

    net = Model(config).cuda()
    if os.path.exists(args.checkpoint):
        pretrained_dict = torch.load(args.checkpoint)
        new_dict = OrderedDict()
        model_dict = net.state_dict()
        for k in pretrained_dict:
            if k in model_dict:
                if pretrained_dict[k].size()==model_dict[k].size():
                    new_dict[k]=pretrained_dict[k]
        model_dict.update(new_dict)
        net.load_state_dict(model_dict)

    if args.phase == 'test':
        test_loader = get_test_loader(config)
        test(net, test_loader, logger)
        save_model(net, save_path=workdir + '/models/EP{}.model'.format(0 + 1))
    elif args.phase == 'train':
        train_loader, test_loader = get_loader(config)
        global global_step
        G_params = set(p for n, p in net.named_parameters() if not n.endswith(".quantiles"))
        optimizer_G = optim.Adam(G_params, lr=config.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_G, milestones=[4000, 4500], gamma=0.1)
        tot_epoch = 5000
        global_step = 0
        best_loss = float("inf")
        steps_epoch = global_step // train_loader.__len__()
        for epoch in range(steps_epoch, tot_epoch):
            logger.info('======Current epoch %s ======' % epoch)
            logger.info(f"Learning rate: {optimizer_G.param_groups[0]['lr']}")
            train_one_epoch(epoch, net, train_loader, optimizer_G, device, logger)
            lr_scheduler.step()

            loss = test(net, test_loader, logger)
            is_best = loss < best_loss
            best_loss = min(loss, best_loss)
            if is_best:
                save_model(net, save_path=workdir + '/models/EP{}_best_loss.model'.format(epoch + 1))
                test(net, test_loader, logger)
            if (epoch + 1) % 2 == 0:
                save_model(net, save_path=workdir + '/models/EP{}.model'.format(epoch + 1))

if __name__ == '__main__':
    main(sys.argv[1:])

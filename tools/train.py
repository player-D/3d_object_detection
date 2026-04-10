import os
import sys
import csv
import random
import argparse
import datetime
import json
import logging
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

# 把项目根目录加入搜索路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tdr_detector import TDRDetector
from models.tdr_loss import TDRLoss
from dataloaders.nuscenes_dataset import NuScenesDataset, collate_fn

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'train.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='TDR-QAF 3D Object Detection Training')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--data_root', type=str, default='./dataset', help='Dataset root path')
    parser.add_argument('--nuscenes_version', type=str, default='v1.0-mini', help='NuScenes version, e.g. v1.0-mini or v1.0-trainval')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of data loading workers')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size') # 降为 2
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate') 
    parser.add_argument('--pretrained', type=str, default='', help='Pre-trained weight .pth file path')
    parser.add_argument('--resume', type=str, default='', help='Resume training from checkpoint .pth file path')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--max_samples', type=int, default=None, help='限制样本数量（用于快速验证）')
    parser.add_argument('--load_indices', type=str, default='', help='加载指定样本索引文件')
    parser.add_argument('--overfit', action='store_true', help='开启本地过拟合测试模式（简化模型）' )
    args = parser.parse_args()
    
    current_time = datetime.datetime.now().strftime("%m_%d_%H-%M")
    save_dir = os.path.join('saved_models', current_time)
    log_dir = os.path.join('logs', current_time)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    logger = setup_logging(log_dir)
    logger.info("==> 硬件环境自检")
    logger.info(f"PyTorch 版本: {torch.__version__}")
    logger.info(f"CUDA 可用: {torch.cuda.is_available()}")
    
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True  # 开启针对固定输入尺寸的硬件加速
    
    config_path = os.path.join(save_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"==> 开始训练，设备: {device}, 总 Epoch: {args.epochs}")
    
    csv_path = os.path.join(log_dir, "train_log.csv")
    with open(csv_path, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['Epoch', 'lr', 'Loss_Total', 'Loss_Cls', 'Loss_Reg', 'Matched_Q', 'Pos_Acc', 'XYZ_Err_m'])
    
    # 初始化 TensorBoard
    writer = SummaryWriter(log_dir=log_dir)
    
    import torch.multiprocessing as mp
    
    try:
        dataset = NuScenesDataset(
            root=args.data_root,
            debug_mode=False,
            max_samples=args.max_samples,
            version=args.nuscenes_version
        )
        logger.info(f"==> 数据集加载成功，共 {len(dataset)} 个样本")
    except Exception as e:
        logger.error(f"==> 加载数据集失败: {e}")
        return
    
    if args.load_indices:
        indices_data = NuScenesDataset.load_sample_indices(args.load_indices)
        dataset.set_sample_indices(indices_data['indices'])
        logger.info(f"==> 已加载样本索引，共 {len(indices_data['indices'])} 个样本")

    sample_indices_path = os.path.join(save_dir, 'sample_indices.json')
    dataset.save_sample_indices(sample_indices_path)
    logger.info(f"==> 训练样本索引已保存到: {sample_indices_path}")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=True if args.num_workers > 0 else False,
        pin_memory=True,
        drop_last=True,
        multiprocessing_context=mp.get_context('spawn') if (torch.cuda.is_available() and args.num_workers > 0) else None
    )

    # 全量配置（服务器标准配置）
    logger.info("==> 🌍 开启全量训练模式：标准配置")
    decoder_layers = 6
    depth_dense = 48
    max_q = 400
    cost_bbox_val = 2.0

    model = TDRDetector(
        num_classes=10 ,
        embed_dims=256 ,
        num_decoder_layers=decoder_layers,
        num_depth_dense=depth_dense,
        max_queries=max_q
    ).to(device)
    
    criterion = TDRLoss(
        num_classes=10 ,      
        cost_cls=1.0 ,       
        cost_bbox=cost_bbox_val
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    start_epoch = 1
    warmup_epochs = 8
    
    # (预训练和断点续训的代码保持原样，篇幅原因这里继续)
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        if 'optimizer_state_dict' in checkpoint: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'epoch' in checkpoint: start_epoch = checkpoint['epoch'] + 1
        if 'scheduler_state_dict' in checkpoint: scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info(f"==> 成功从 checkpoint 恢复训练")

    best_xyz_err = float('inf')  # 记录最优指标
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            
            # Warmup + Scheduler
            if epoch <= warmup_epochs:
                current_lr = args.lr * (epoch / warmup_epochs)
                for pg in optimizer.param_groups:
                    pg['lr'] = current_lr
            else:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]

            model.train()
            total_loss, total_cls_loss, total_reg_loss = 0.0, 0.0, 0.0
            total_matched, total_pos_acc, total_xyz_err = 0.0, 0.0, 0.0
            steps = len(dataloader)
            actual_steps = 0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}", mininterval=0.5, dynamic_ncols=True)
            for batch_idx, (images, boxes_2d, cam_intrinsics, cam_extrinsics, gt_bboxes, gt_labels) in enumerate(pbar):
                
                # ================== 【双保险：防空批次暴毙】 ==================
                if images.numel() == 0 or images.dim() == 1:   # 更鲁棒的写法
                    tqdm.write("⚠️  遭遇全损空 Batch，已安全跳过！")
                    continue
                # ============================================================
                
                actual_steps += 1  # 只处理有效batch时才计数
                
                images = images.to(device).float()
                boxes_2d = boxes_2d.to(device).float()
                cam_intrinsics = cam_intrinsics.to(device).float()
                cam_extrinsics = cam_extrinsics.to(device).float()
                gt_labels = [label.to(device) for label in gt_labels]
                gt_bboxes = [bbox.to(device) for bbox in gt_bboxes]
                
                optimizer.zero_grad()
                
                # 全精度前向传播
                cls_scores, bbox_preds = model(images, boxes_2d, cam_intrinsics, cam_extrinsics, temporal_depth_prior=None)
                losses = criterion(cls_scores, bbox_preds, gt_labels, gt_bboxes)
                loss_cls = losses['loss_cls']
                loss_reg = losses['loss_bbox']
                loss = loss_cls + loss_reg
                
                # 防爆保险丝
                if torch.isnan(loss) or torch.isinf(loss):
                    tqdm.write(f"⚠️ 发现 NaN Loss，主动丢弃当前有毒 Batch！")
                    optimizer.zero_grad()
                    continue
                    
                loss.backward()
                
                # 【修复】：先清 NaN，再裁剪梯度！
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
                optimizer.step()
                
                matched_q = losses.get('matched_queries', 0)
                pos_acc = losses.get('pos_acc', 0)
                xyz_err = losses.get('xyz_err_m', 0)
                
                total_loss += loss.item()
                total_cls_loss += loss_cls.item()
                total_reg_loss += loss_reg.item()
                total_matched += matched_q
                total_pos_acc += pos_acc
                total_xyz_err += xyz_err
                
                pbar.set_postfix({
                    'LR': f'{current_lr:.6f}',
                    'Loss': f'{loss.item():.2f}',
                    'Match': f'{matched_q}',
                    'Acc': f'{pos_acc*100:.1f}%',
                    'Err': f'{xyz_err:.2f}m'
                })
            
            denom = max(1, actual_steps)
            avg_loss = total_loss / denom
            avg_matched = total_matched / denom
            avg_acc = total_pos_acc / denom
            avg_err = total_xyz_err / denom
            
            with open(csv_path, 'a', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([epoch, current_lr, avg_loss, total_cls_loss/denom, total_reg_loss/denom, avg_matched, avg_acc, avg_err])
            
            logger.info(f"==> Epoch {epoch}: LR={current_lr:.6f}, Loss={avg_loss:.2f}, Match={avg_matched:.1f}, Acc={avg_acc*100:.1f}%, XYZ_Err={avg_err:.2f}m")
            
            # 在每个 epoch 结束打印日志之后，写入 TensorBoard
            writer.add_scalar('Loss/Total', avg_loss, epoch)
            writer.add_scalar('Loss/Cls', total_cls_loss/denom, epoch)
            writer.add_scalar('Loss/Reg', total_reg_loss/denom, epoch)
            writer.add_scalar('Metrics/Pos_Acc', avg_acc, epoch)
            writer.add_scalar('Metrics/XYZ_Err', avg_err, epoch)
            
            # 【自动保存最优权重】
            if avg_err < best_xyz_err:
                best_xyz_err = avg_err
                best_model_path = os.path.join(save_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                }, best_model_path)
                logger.info(f"🏆 发现更优模型！已保存至 {best_model_path}")
            
            model_save_path = os.path.join(save_dir, f'tdr_qaf_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }, model_save_path)
            logger.info(f"==> 模型已保存至: {model_save_path}")
    except KeyboardInterrupt:
        logger.info("\n⚠️ 训练被手动中断 (Ctrl+C)，正在安全退出...")
    except Exception as e:
        logger.error(f"❌ 训练异常: {e}")
    finally:
        writer.close()

if __name__ == '__main__':
    main()

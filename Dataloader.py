import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from PIL import Image, ImageFile

from torch.utils.data import Dataset
import os.path as osp
import random
import torch
import numpy as np
import math

from timm.data.random_erasing import RandomErasing
from utility import RandomIdentitySampler,RandomErasing3
from Datasets.MARS_dataset import Mars
from Datasets.iLDSVID import iLIDSVID
from Datasets.PRID_dataset import PRID

__factory = {
    'Mars':Mars,
    'iLIDSVID':iLIDSVID,
    'PRID':PRID
}

def train_collate_fn(batch):
    w = zip(*batch)
    # print("len(w)", len(w))
    imgs, audios, pids, camids, a = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)

    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), torch.stack(audios, dim=0), pids, camids, torch.stack(a, dim=0)

def val_collate_fn(batch):
    
    imgs, pids, camids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids_batch,  img_paths

def dataloader(Dataset_name):
    train_transforms = T.Compose([
            T.Resize([256, 128], interpolation=3),
            T.RandomHorizontalFlip(p=0.5),
            T.Pad(10),
            T.RandomCrop([256, 128]),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            
            
        ])

    val_transforms = T.Compose([
        T.Resize([256, 128]),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    

    dataset = __factory[Dataset_name]()
    train_set = VideoDataset_inderase(dataset.train, seq_len=6, sample='intelligent',transform=train_transforms)
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

   
    train_loader = DataLoader(train_set, batch_size=32,sampler=RandomIdentitySampler(dataset.train, 64,4),num_workers=4, collate_fn=train_collate_fn)
  
    q_val_set = VideoDataset(dataset.query, seq_len=6, sample='dense', transform=val_transforms)
    g_val_set = VideoDataset(dataset.gallery, seq_len=6, sample='dense', transform=val_transforms)
    
    
    return train_loader, len(dataset.query), num_classes, cam_num, view_num,q_val_set,g_val_set



def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

def read_audio_npy(npy_path):
    """
    Keep reading npy file until succeed.
    This can avoid IOError incurred by heavy IO process.
    The function reads .npy audio file and converts it to a PyTorch tensor.
    """
    got_data = False
    while not got_data:
        try:
            # 读取npy文件
            data = np.load(npy_path)
            # 将数据转换为PyTorch张量
            tensor = torch.tensor(data, dtype=torch.float32)
            got_data = True
        except IOError:
            print(f"IOError incurred when reading '{npy_path}'. Will redo. Don't worry. Just chill.")
        except ValueError as e:
            print(f"Error processing '{npy_path}': {e}")
            break
    return tensor

class VideoDataset(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, seq_len=15, sample='evenly', transform=None , max_length=40):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)

        # if self.sample == 'restricted_random':
        #     frame_indices = range(num)
        #     chunks = 
        #     rand_end = max(0, len(frame_indices) - self.seq_len - 1)
        #     begin_index = random.randint(0, rand_end)


        if self.sample == 'random':
            """
            Randomly sample seq_len consecutive frames from num frames,
            if num is smaller than seq_len, then replicate items.
            This sampling strategy is used in training phase.
            """
            frame_indices = range(num)
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))

            indices = frame_indices[begin_index:end_index]
            # print(begin_index, end_index, indices)
            if len(indices) < self.seq_len:
                indices=np.array(indices)
                indices = np.append(indices , [indices[-1] for i in range(self.seq_len - len(indices))])
            else:
                indices=np.array(indices)
            imgs = []
            targt_cam=[]
            for index in indices:
                index=int(index)
                img_path = img_paths[index]
                img = read_image(img_path)
                if self.transform is not None:
                    img = self.transform(img)
                img = img.unsqueeze(0)
                targt_cam.append(camid)
                imgs.append(img)
            imgs = torch.cat(imgs, dim=0)
            #imgs=imgs.permute(1,0,2,3)
            return imgs, pid, targt_cam

        elif self.sample == 'dense':
            """
            Sample all frames in a video into a list of clips, each clip contains seq_len frames, batch_size needs to be set to 1.
            This sampling strategy is used in test phase.
            """
            # import pdb
            # pdb.set_trace()
        
            cur_index=0
            frame_indices = [i for i in range(num)]
            indices_list=[]
            while num-cur_index > self.seq_len:
                indices_list.append(frame_indices[cur_index:cur_index+self.seq_len])
                cur_index+=self.seq_len
            last_seq=frame_indices[cur_index:]
            # print(last_seq)
            for index in last_seq:
                if len(last_seq) >= self.seq_len:
                    break
                last_seq.append(index)
            indices_list.append(last_seq)
            
#             indices_list = []
#             if num >= self.seq_len:
#                 r = num % self.seq_len
#                 stride = num // self.seq_len
#                 if r != 0:
#                     stride += 1
#                 for i in range(stride):
#                     indices = np.arange(i, stride * self.seq_len, stride)
#                     indices = indices.clip(max=num - 1)
#                     indices_list.append(indices)

#             else:
#                 # 如果 num 小于 seq_len，复制最后一帧直到满足 seq_len
#                 indices = np.arange(0, num)
#                 num_pads = self.seq_len - num
#                 indices = np.concatenate([indices, np.ones(num_pads).astype(np.int32) * (num - 1)])
#                 indices_list.append(indices)
#             if len(indices_list) > 50:
#                 indices_list = indices_list[:50]
            
            
            imgs_list=[]
            audios_list = []
            targt_cam=[]
            # print(indices_list , num , img_paths  )
            for indices in indices_list:
                if len(imgs_list) > self.max_length:
                    break 
                imgs = []
                audios = []
                for index in indices:
                    index=int(index)
                    img_path = img_paths[index]
                    
                    # if 'cam_a' in img_path:
                    #     audio_path = img_path.replace('cam_a', 'cam_a_audio_npy').replace('.png', '.npy')
                    # elif 'cam_b' in img_path:
                    #     audio_path = img_path.replace('cam_b', 'cam_b_audio_npy').replace('.png', '.npy')
                    
                    if 'bbox_train' in img_path:
                        audio_path = img_path.replace('bbox_train', 'bbox_train_audio_npy').replace('.jpg', '.npy')
                    elif 'bbox_test' in img_path:
                        audio_path = img_path.replace('bbox_test', 'bbox_test_audio_npy').replace('.jpg', '.npy')
                    
                    # print("audio_path", audio_path)
                    img = read_image(img_path)
                    audio = read_audio_npy(audio_path)
                    # print("audio.shape". audio.shape)
                    
                    if self.transform is not None:
                        img = self.transform(img)
                    img = img.unsqueeze(0)
                    imgs.append(img)
                    audios.append(audio)
                    targt_cam.append(camid)
                    
                imgs = torch.cat(imgs, dim=0)
                audios = torch.cat(audios, dim=0)
                
                #imgs=imgs.permute(1,0,2,3)
                
                imgs_list.append(imgs)
                audios_list.append(audios)
            imgs_array = torch.stack(imgs_list)
            audios_array = torch.stack(audios_list)
            return imgs_array, audios_array, pid, targt_cam, img_paths
            # return imgs_array, pid, targt_cam,img_paths
            #return imgs_array, pid, int(camid),trackid


        elif self.sample == 'dense_subset':
            """
            Sample all frames in a video into a list of clips, each clip contains seq_len frames, batch_size needs to be set to 1.
            This sampling strategy is used in test phase.
            """
            frame_indices = range(num)
            rand_end = max(0, len(frame_indices) - self.max_length - 1)
            begin_index = random.randint(0, rand_end)
            

            cur_index=begin_index
            frame_indices = [i for i in range(num)]
            indices_list=[]
            while num-cur_index > self.seq_len:
                indices_list.append(frame_indices[cur_index:cur_index+self.seq_len])
                cur_index+=self.seq_len
            last_seq=frame_indices[cur_index:]
            # print(last_seq)
            for index in last_seq:
                if len(last_seq) >= self.seq_len:
                    break
                last_seq.append(index)
            

            indices_list.append(last_seq)
            imgs_list=[]
            # print(indices_list , num , img_paths  )
            for indices in indices_list:
                if len(imgs_list) > self.max_length:
                    break 
                imgs = []
                for index in indices:
                    index=int(index)
                    img_path = img_paths[index]
                    img = read_image(img_path)
                    if self.transform is not None:
                        img = self.transform(img)
                    img = img.unsqueeze(0)
                    imgs.append(img)
                imgs = torch.cat(imgs, dim=0)
                #imgs=imgs.permute(1,0,2,3)
                imgs_list.append(imgs)
            imgs_array = torch.stack(imgs_list)
            return imgs_array, pid, camid
        
        elif self.sample == 'intelligent_random':
            # frame_indices = range(num)
            indices = []
            each = max(num//seq_len,1)
            for  i in range(seq_len):
                if i != seq_len -1:
                    indices.append(random.randint(min(i*each , num-1), min( (i+1)*each-1, num-1)) )
                else:
                    indices.append(random.randint(min(i*each , num-1), num-1) )
            print(len(indices))
            imgs = []
            for index in indices:
                index=int(index)
                img_path = img_paths[index]
                img = read_image(img_path)
                if self.transform is not None:
                    img = self.transform(img)
                img = img.unsqueeze(0)
                imgs.append(img)
            imgs = torch.cat(imgs, dim=0)
            #imgs=imgs.permute(1,0,2,3)
            return imgs, pid, camid
        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))

            

        
class VideoDataset_inderase(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, seq_len=15, sample='evenly', transform=None , max_length=40):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
        self.max_length = max_length
        self.erase = RandomErasing3(probability=0.5, mean=[0.485, 0.456, 0.406])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)
        if self.sample != "intelligent":
            frame_indices = range(num)
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))

            indices1 = frame_indices[begin_index:end_index]
            indices = []
            for index in indices1:
                if len(indices1) >= self.seq_len:
                    break
                indices.append(index)
            indices=np.array(indices)
        else:
            # frame_indices = range(num)
            indices = []
            each = max(num//self.seq_len,1)
            for  i in range(self.seq_len):
                if i != self.seq_len -1:
                    indices.append(random.randint(min(i*each , num-1), min( (i+1)*each-1, num-1)) )
                else:
                    indices.append(random.randint(min(i*each , num-1), num-1) )
            # print(len(indices), indices, num )
        imgs = []
        audios = []
        labels = []
        targt_cam=[]
        
        for index in indices:
            index=int(index)
            img_path = img_paths[index]
            
            # if 'cam_a' in img_path:
            #     audio_path = img_path.replace('cam_a', 'cam_a_audio_npy').replace('.png', '.npy')
            # elif 'cam_b' in img_path:
            #     audio_path = img_path.replace('cam_b', 'cam_b_audio_npy').replace('.png', '.npy')
            if 'bbox_train' in img_path:
                audio_path = img_path.replace('bbox_train', 'bbox_train_audio_npy').replace('.jpg', '.npy')
            elif 'bbox_test' in img_path:
                audio_path = img_path.replace('bbox_test', 'bbox_test_audio_npy').replace('.jpg', '.npy')
            
            # print("audio_path", audio_path)
            img = read_image(img_path)
            audio = read_audio_npy(audio_path)
            # print("audio.shape", audio.shape)
            
            if self.transform is not None:
                img = self.transform(img)
            img , temp  = self.erase(img)
            labels.append(temp)
            img = img.unsqueeze(0)
            # print("img.shape", img.shape)
            # print("audio.shape", audio.shape)
            imgs.append(img)
            targt_cam.append(camid)
            audios.append(audio)
            
        labels = torch.tensor(labels)
        imgs = torch.cat(imgs, dim=0)
        audios = torch.cat(audios, dim=0)
        
        # return imgs, pid, targt_cam ,labels
        return imgs, audios, pid, targt_cam, labels
        


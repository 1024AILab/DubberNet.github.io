# Harnessing the Power of Audio-Visual Collaboration for Video-based Person ReID

> **Project Home** for our paper: *Harnessing the Power of Audio-Visual Collaboration for Video-based Person ReID*  
> [Paper] | [Project Page] | [Dataset (TED)]

---

## 🌟 Highlights
- We propose a noval Dubbing-based audio-visual collaborative framework named DubberNet for video-based person ReID. To the best of our knowledge, we are the first to introduce audio for advancing this field.
- We propose an efficient multimodal learning and transfer mechanism that employs a Simple Injection (SI) module to efficiently fuse audio and a structurally consistent Simple Perception (SP) module to perceive knowledge.
- We propose a token-wise selective distillation strategy, in which a mask generator (MG) adaptively preserves informative tokens, allowing the student to inherit more robust knowledge from the teacher.
- Extensive experiments demonstrates that DubberNet outperforms the state-of-the-art methods on three standard benchmarks (MARS, iLIDS-VID, and PRID), achieving Rank-1 accuracy of 97.0% and 96.0% on MARS and iLIDS-VID, respectively.


---

## 🖼️ Teaser
<p align="center">
  <img src="asset/f3.png" alt="Method Overview" width="720">
</p>

---
## 🎧 TED Audio-Visual Mini Gallery

<table>
  <tr>
    <td align="center">
  <figure>
    <!-- 6 images: 2 rows × 3 columns, each 256×128 -->
    <p>
      <img src="./asset/ted_samples/002/002.gif" alt="TED sample #1-1" width="256" height="128">
    </p>
    <figcaption><sub>Sample Group #1 · PID 002 </sub></figcaption>
    <!-- one audio for the whole group -->
    <audio controls preload="none">
      <source src="./asset/ted_samples/002/002.wav" type="audio/wav">
      Your browser does not support the audio element.
    </audio>
  </figure>
</td>
    <td align="center">
  <figure>
    <!-- 6 images: 2 rows × 3 columns, each 256×128 -->
    <p>
      <img src="./asset/ted_samples/008/008.gif" alt="TED sample #1-1" width="256" height="128">
    </p>
    <figcaption><sub>Sample Group #2 · PID 008 </sub></figcaption>
    <!-- one audio for the whole group -->
    <audio controls preload="none">
      <source src="./asset/ted_samples/008/008.wav" type="audio/wav">
      Your browser does not support the audio element.
    </audio>
  </figure>
</td>
    <td align="center">
  <figure>
    <!-- 6 images: 2 rows × 3 columns, each 256×128 -->
    <p>
      <img src="./asset/ted_samples/013/013.gif" alt="TED sample #1-3" width="256" height="128">
    </p>
    <figcaption><sub>Sample Group #3 · PID 013 </sub></figcaption>
    <!-- one audio for the whole group -->
    <audio controls preload="none">
      <source src="./asset/ted_samples/008/008.wav" type="audio/wav">
      Your browser does not support the audio element.
    </audio>
  </figure>
</td>
  </tr>
<tr>
    <td align="center">
  <figure>
    <!-- 6 images: 2 rows × 3 columns, each 256×128 -->
    <p>
      <img src="./asset/ted_samples/032/032.gif" alt="TED sample #1-3" width="256" height="128">
    </p>
    <figcaption><sub>Sample Group #4 · PID 032 </sub></figcaption>
    <!-- one audio for the whole group -->
    <audio controls preload="none">
      <source src="./asset/ted_samples/032/032.wav" type="audio/wav">
      Your browser does not support the audio element.
    </audio>
  </figure>
</td>
    <td align="center">
  <figure>
    <!-- 6 images: 2 rows × 3 columns, each 256×128 -->
    <p>
      <img src="./asset/ted_samples/034/034.gif" alt="TED sample #1-3" width="256" height="128">
    </p>
    <figcaption><sub>Sample Group #5 · PID 034 </sub></figcaption>
    <!-- one audio for the whole group -->
    <audio controls preload="none">
      <source src="./asset/ted_samples/034/034.wav" type="audio/wav">
      Your browser does not support the audio element.
    </audio>
  </figure>
</td>
    <td align="center">
  <figure>
    <!-- 6 images: 2 rows × 3 columns, each 256×128 -->
    <p>
      <img src="./asset/ted_samples/037/037.gif" alt="TED sample #1-3" width="256" height="128">
    </p>
    <figcaption><sub>Sample Group #6 · PID 037 </sub></figcaption>
    <!-- one audio for the whole group -->
    <audio controls preload="none">
      <source src="./asset/ted_samples/037/037.wav" type="audio/wav">
      Your browser does not support the audio element.
    </audio>
  </figure>
</td>
</tr>
</table>

---
## 🔊 Dubbing Dataset Mini Gallery (Audio Only)

<table>
  <tr>
    <td></td>
    <td align="center"><b>PID 0001</b></td>
    <td></td>
  </tr>
  <tr>
    <td align="center">
      <audio controls preload="none" style="width:256px;">
        <source src="./asset/Mars/0001/1.mp3" type="audio/wav">
        Your browser does not support the audio element.
      </audio>
    </td>
    <td align="center">
      <audio controls preload="none" style="width:256px;">
        <source src="./asset/Mars/0001/2.mp3" type="audio/wav">
        Your browser does not support the audio element.
      </audio>
    </td>
    <td align="center">
      <audio controls preload="none" style="width:256px;">
        <source src="./asset/Mars/0001/3.mp3" type="audio/wav">
        Your browser does not support the audio element.
      </audio>
    </td>
  </tr>
</table>
<table>
   <tr>
    <td></td>
    <td align="center"><b>PID 0003</b></td>
    <td></td>
  </tr>
  <tr>
    <td align="center">
      <audio controls preload="none" style="width:256px;">
        <source src="./asset/Mars/0003/1.mp3" type="audio/wav">
        Your browser does not support the audio element.
      </audio>
    </td>
    <td align="center">
      <audio controls preload="none" style="width:256px;">
        <source src="./asset/Mars/0003/2.mp3" type="audio/wav">
        Your browser does not support the audio element.
      </audio>
    </td>
    <td align="center">
      <audio controls preload="none" style="width:256px;">
        <source src="./asset/Mars/0003/3.mp3" type="audio/wav">
        Your browser does not support the audio element.
      </audio>
    </td>
  </tr>
</table>
<table>
   <tr>
    <td></td>
    <td align="center"><b>PID 0005</b></td>
    <td></td>
  </tr>
  <tr>
    <td align="center">
      <audio controls preload="none" style="width:256px;">
        <source src="./asset/Mars/0005/1.mp3" type="audio/wav">
        Your browser does not support the audio element.
      </audio>
    </td>
    <td align="center">
      <audio controls preload="none" style="width:256px;">
        <source src="./asset/Mars/0005/2.mp3" type="audio/wav">
        Your browser does not support the audio element.
      </audio>
    </td>
    <td align="center">
      <audio controls preload="none" style="width:256px;">
        <source src="./asset/Mars/0005/3.mp3" type="audio/wav">
        Your browser does not support the audio element.
      </audio>
    </td>
  </tr>
</table>


---

## 📦 Installation
```bash
conda create -n avreid python=3.10 -y
conda activate avreid
pip install -r requirements.txt
```

---

## 📁 Data Preparation (TED)
```
data/
└── TED/
    ├── train/
    ├── query/              
    ├── gallery/
    ├── train_audio/
    ├── query_audio/
    └── gallery_audio/              
```

---

## 🚀 Quick Start
```bash
# train
python train.py --cfg configs/av_reid_ted.yaml
```

---

## 📊 Results
| Method       | Venue      | PRID R1 | PRID R5 | iLIDS R1 | iLIDS R5 | MARS R1 | MARS mAP |
|--------------|------------|:-------:|:-------:|:--------:|:--------:|:-------:|:--------:|
| SCAN         | TIP 2019   | 95.3    | 99.0    | 88.0     | 96.7     | 87.2    | 77.2     |
| AdaGraph     | TIP 2020   | 94.6    | 99.1    | 84.5     | 96.7     | 89.5    | 81.9     |
| MGH          | CVPR 2020  | 94.8    | 99.3    | 85.6     | 97.1     | 90.0    | 85.8     |
| MG-RAFA      | CVPR 2020  | 95.9    | 99.7    | 88.6     | 98.0     | 88.8    | 85.9     |
| STRF         | ICCV 2021  | -       | -       | 89.3     | -        | 90.3    | 86.1     |
| PhD          | CVPR 2021  | 96.6    | 97.8    | -        | -        | 88.9    | 86.2     |
| CTL          | CVPR 2021  | -       | -       | 89.7     | 97.0     | 91.4    | 86.7     |
| MFA          | TIP 2022   | 95.5    | 100.0   | 93.3     | 99.3     | 90.4    | 85.0     |
| MSTAT        | TMM 2022   | -       | -       | 93.3     | 99.3     | 91.8    | 85.3     |
| AGW          | TPAMI 2022 | 94.4    | 98.4    | 83.2     | 98.3     | 87.6    | 83.0     |
| SGMN         | TCSVT 2022 | -       | -       | 88.6     | 96.7     | 90.8    | 85.3     |
| CAViT        | ECCV 2022  | 95.5    | 98.9    | 93.3     | 98.0     | 90.8    | 87.2     |
| SINet        | CVPR 2022  | -       | -       | 92.5     | -        | 91.0    | 86.2     |
| FIND         | TIP 2023   | -       | -       | 91.3     | 98.0     | 91.4    | 86.8     |
| RGCN         | TCSVT 2023 | -       | -       | 90.2     | 98.5     | 91.1    | 86.5     |
| STFE         | TMM 2024   | -       | -       | -        | -        | 94.5    | 89.5     |
| HASI         | TCSVT 2024 | 96.1    | 98.7    | 93.3     | 99.6     | 91.4    | 87.5     |
| TMT          | TITS 2024  | -       | -       | 91.3     | 98.6     | 91.8    | 86.5     |
| TF-CLIP      | AAAI 2024  | -       | -       | 94.5     | 99.1     | 93.0    | 89.4     |
| TCViT        | AAAI 2024  | -       | -       | 94.3     | 99.3     | 91.7    | 87.6     |
| 3DAPRL       | TCSVT 2025 | 96.6    | 98.9    | 94.7     | 98.7     | 93.1    | 90.3     |
| PCPT         | TIP 2025   | 96.4    | -       | 92.9     | -        | 90.9    | 86.6     |
| DS-VReID     | IJCV 2025  | 95.5    | -       | 94.0     | -        | 92.3    | 87.6     |
| **DubberNet-S** | Ours    | **96.6**| **100.0**| **96.0** | **100.0**| **97.0**| **90.5**  |

> **Table 1.** Comparison with state-of-the-art video-based person ReID methods on PRID, iLIDS-VID, and MARS datasets. **Bold indicates the best.**

---

## 📜 Citation
```bibtex
@inproceedings{YourAVReID2025,
  title={Harnessing the Power of Audio-Visual Collaboration for Video-based Person ReID},
  author={You, A. and Collaborators, B.},
  booktitle={Proceedings of ...},
  year={2025}
}
```

---

## 📄 License
This repo is released under the MIT License (see `LICENSE`).

---

## 🤝 Acknowledgements
We thank the contributors and dataset providers (including TED) for their support.

---

## 📧 Contact
- Maintainer: your.name (youremail@example.com)
- Issues: please open a GitHub Issue

# Harnessing the Power of Audio-Visual Collaboration for Video-based Person ReID

> **Project Home** for our paper: *Harnessing the Power of Audio-Visual Collaboration for Video-based Person ReID*  
> [Paper] | [Project Page] | [Dataset (TED)]



## 📚 Table of Contents

- [🌟 Highlights](#highlights)
- [🖼️ Teaser](#teaser)
- [🎧 TED Audio-Visual Mini Gallery](#ted-audio-visual-mini-gallery)
- [🔊 Dubbing Dataset Mini Gallery (Audio Only)](#dubbing-dataset-mini-gallery-audio-only)
- [📦 Installation](#installation)
- [📁 Data Preparation](#data-preparation)
- [🚀 Quick Start](#quick-start)
- [📊 Results](#results)



---

<a id="highlights"></a>

## 🌟 Highlights
- We propose a noval Dubbing-based audio-visual collaborative framework named DubberNet for video-based person ReID. To the best of our knowledge, we are the first to introduce audio for advancing this field.
- We propose an efficient multimodal learning and transfer mechanism that employs a Simple Injection (SI) module to efficiently fuse audio and a structurally consistent Simple Perception (SP) module to perceive knowledge.
- We propose a token-wise selective distillation strategy, in which a mask generator (MG) adaptively preserves informative tokens, allowing the student to inherit more robust knowledge from the teacher.
- Extensive experiments demonstrates that DubberNet outperforms the state-of-the-art methods on three standard benchmarks (MARS, iLIDS-VID, and PRID), achieving Rank-1 accuracy of 97.0% and 96.0% on MARS and iLIDS-VID, respectively.


---

<a id="teaser"></a>
## 🖼️ Teaser
<p align="center">
  <img src="asset/f3.png" alt="Method Overview" width="720">
</p>

---
<a id="ted-audio-visual-mini-gallery"></a>
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
<a id="dubbing-dataset-mini-gallery-audio-only"></a>
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

<a id="installation"></a>
## 📦 Installation
```bash
conda create -n avreid python=3.10 -y
conda activate avreid
pip install -r requirements.txt
```



---

<a id="data-preparation"></a>
# 📁 Data Preparation

## 1) Download Common Voice Scripted Speech 24.0 (English)
1. Download **Common Voice Scripted Speech 24.0 – English** from:  
   https://datacollective.mozillafoundation.org/datasets/cmj8u3p1w0075nxxbe8bedl00
2. Extract the archive. After extraction, you should have (at least):
   - `.../en/train.tsv`
   - `.../clips/` (audio files)

> We use `train.tsv` + `clips/` to allocate audio to each identity in the ReID datasets. Specifically, we index each entry by `client_id` and link it to the corresponding audio file under `clips/` via the `path` field (i.e., the audio filename); an example row from `train.tsv` is shown below.
> ```text client_id	path	sentence	up_votes	down_votes	age	gender	accent	locale	segment f1f6414c04e74453065e1b7fc1639c6f728dc03ed9589034b8531d8c7d8b994f223f2b79d5fcc42a2b7b19f8cbca5af08f31d47a554ddd682df04ba62caaaa56 common_voice_en_20009651.mp3	"It just didn't seem fair."	2	1				en	```

---

## 2) Update dataset paths in `audio_allocate/`
Before running audio allocation, update the paths in the scripts below.  
Replace all `xxx` with your **absolute local path**.

---

## ✅ MARS / PRID / iLIDS-VID (Path Configuration)

### MARS
**Edit:** `audio_allocate/Mars.py`

```python
train_path = r'xxx\bbox_train'
test_path  = r'xxx\Mars\bbox_test'

tsv_paths = [
    (r'xxx\en\train.tsv', r'xxx\clips'),
]

train_audio_path = r'xxx\Mars\bbox_train_audio'
test_audio_path  = r'xxx\Mars\bbox_test_audio'
```

**Path meanings**
- `train_path`: MARS training set image directory (`bbox_train`)
- `test_path`: MARS test set image directory (`bbox_test`)
- `tsv_paths`: Common Voice English `train.tsv` and corresponding `clips/` directory
- `train_audio_path`: output directory for audio-augmented training set
- `test_audio_path`: output directory for audio-augmented test set

---

### PRID (multi-shot)
**Edit:** `audio_allocate/PRID.py`

```python
cam1_path = r'xxx\multi_shot\cam_a'
cam2_path = r'xxx\multi_shot\cam_b'

tsv_path = r'xxx\train.tsv'
audio_clips_path = r'xxx\clips'

cam1_audio_path = r'xxx\multi_shot\cam_a_audio'
cam2_audio_path = r'xxx\multi_shot\cam_b_audio'
```

**Path meanings**
- `cam1_path`: PRID camera A image directory (`multi_shot/cam_a`)
- `cam2_path`: PRID camera B image directory (`multi_shot/cam_b`)
- `tsv_path`: Common Voice `train.tsv`
- `audio_clips_path`: Common Voice `clips/` directory
- `cam1_audio_path`: output directory for camera A with allocated audio
- `cam2_audio_path`: output directory for camera B with allocated audio

---

### iLIDS-VID
**Edit:** `audio_allocate/iLIDS-VID.py`

```python
cam1_path = r'xxx\sequences\cam1'
cam2_path = r'xxx\sequences\cam2'

tsv_path = r'xxx\train.tsv'
audio_clips_path = r'xxx\clips'

cam1_audio_path = r'xxx\sequences\cam1_audio'
cam2_audio_path = r'xxx\sequences\cam2_audio'
```

**Path meanings**
- `cam1_path`: iLIDS-VID camera 1 sequences directory (`sequences/cam1`)
- `cam2_path`: iLIDS-VID camera 2 sequences directory (`sequences/cam2`)
- `tsv_path`: Common Voice `train.tsv`
- `audio_clips_path`: Common Voice `clips/` directory
- `cam1_audio_path`: output directory for camera 1 with allocated audio
- `cam2_audio_path`: output directory for camera 2 with allocated audio






---

<a id="quick-start"></a>
## 🚀 Quick Start

If you want to train the model on the TED dataset, follow the instruction in [TED](https://github.com/1024AILab/TED.git).

```bash
# To train the teacher model run:
python -u VID_Trans_ReID.py --Dataset_name 'PRID'
# To train the student model run:
python -u VID_KD_demo.py --Dataset_name 'PRID'
# To train the teacher model run:
python -u VID_Trans_ReID.py --Dataset_name 'Mars'
# To train the student model run:
python -u VID_KD_demo.py --Dataset_name 'Mars'
```

---

<a id="results"></a>
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

<a id="citation"></a>
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

<a id="license"></a>
## 📄 License
This repo is released under the MIT License (see `LICENSE`).

---

<a id="acknowledgements"></a>
## 🤝 Acknowledgements
We thank the contributors and dataset providers (including TED) for their support.

---

<a id="contact"></a>
## 📧 Contact
- Maintainer: your.name (youremail@example.com)
- Issues: please open a GitHub Issue

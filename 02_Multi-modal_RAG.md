# Multi-modal RAG: Text + Image + Audio

## 1. Giới thiệu

### 1.1. Giới hạn của RAG truyền thống

RAG cơ bản chỉ xử lý **text**. Nhưng thực tế, thông tin tồn tại ở nhiều dạng:
- **Images**: Biểu đồ, sơ đồ, ảnh sản phẩm, scan tài liệu
- **Audio**: Ghi âm cuộc họp, podcast, bài giảng
- **Video**: Webinar, tutorial, demo

### 1.2. Multi-modal RAG là gì?

Hệ thống có khả năng:
1. Index nhiều loại media (text, image, audio, video)
2. Retrieve thông tin từ đa phương tiện
3. Sinh câu trả lời dựa trên context đa dạng

**Ví dụ use case**:
- Chatbot hỗ trợ kỹ thuật: "Làm sao sửa lỗi này?" → Trả về ảnh minh họa + hướng dẫn text
- Tìm kiếm học liệu: "Giải thích về photosynthesis" → Trả về video + diagram + text
- Phân tích meeting: "John nói gì về budget?" → Tìm đoạn audio + transcript

---

## 2. Kiến trúc Multi-modal RAG

### 2.1. Tổng quan pipeline

```
┌─────────────┐
│   PDF       │──┐
│   Images    │  │
│   Audio     │  ├──→ Extractors ──→ Embeddings ──→ Vector DB
│   Video     │  │
└─────────────┘──┘
                          ↓
User Query ──→ Multi-modal Retrieval ──→ LLM (GPT-4V / Gemini) ──→ Answer
```

### 2.2. Thành phần chính

1. **Media Extractors**:
   - Text: PyPDF, Markdown parser
   - Image: OCR (Tesseract), Object Detection
   - Audio: Whisper (Speech-to-Text)
   - Video: Frame extraction + Audio extraction

2. **Multi-modal Embeddings**:
   - CLIP (text-image joint embedding)
   - ImageBind (7 modalities)
   - Gemini Embeddings (native multi-modal)

3. **Vector Database**:
   - Chroma, Qdrant, Weaviate (hỗ trợ metadata)

4. **Multi-modal LLM**:
   - GPT-4 Vision
   - Gemini 1.5 / 2.0
   - LLaVA (open-source)

---

## 3. Xử lý từng loại Media

### 3.1. Image Processing

#### a) OCR cho ảnh chứa text

```python
# image_processing.py
import pytesseract
from PIL import Image

def extract_text_from_image(image_path):
    """Extract text từ ảnh scan, screenshot"""
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang='vie')  # Tiếng Việt
    return text

# Ví dụ
text = extract_text_from_image("lecture_slide.png")
print(text)
```

#### b) Image Captioning

Sinh mô tả tự động cho ảnh không có text:

```python
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image_path):
    img = Image.open(image_path)
    inputs = processor(img, return_tensors="pt")
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

# Ví dụ
caption = generate_caption("product_photo.jpg")
# Output: "a laptop computer on a wooden desk"
```

#### c) CLIP Embeddings (Text + Image cùng space)

```python
from transformers import CLIPProcessor, CLIPModel
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def embed_image(image_path):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_embedding = model.get_image_features(**inputs)
    return image_embedding.numpy()[0]

def embed_text(text):
    inputs = processor(text=text, return_tensors="pt")
    with torch.no_grad():
        text_embedding = model.get_text_features(**inputs)
    return text_embedding.numpy()[0]

# Cùng embedding space → có thể tìm ảnh bằng text query
img_vec = embed_image("chart.png")
query_vec = embed_text("sales chart")
# Similarity: cosine(img_vec, query_vec)
```

### 3.2. Audio Processing

#### a) Speech-to-Text với Whisper

```python
# audio_processing.py
import whisper

model = whisper.load_model("base")  # tiny, base, small, medium, large

def transcribe_audio(audio_path):
    result = model.transcribe(audio_path, language="vi")  # Tiếng Việt
    return result["text"]

# Ví dụ
transcript = transcribe_audio("meeting_recording.mp3")
print(transcript)
```

#### b) Diarization (phân biệt speaker)

```python
from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

def diarize_audio(audio_path):
    diarization = pipeline(audio_path)
    
    speakers = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speakers.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })
    return speakers

# Kết hợp với Whisper
def transcribe_with_speakers(audio_path):
    # 1. Diarization
    speakers = diarize_audio(audio_path)
    
    # 2. Transcribe
    full_transcript = transcribe_audio(audio_path)
    
    # 3. Align (đơn giản hóa)
    result = []
    for segment in speakers:
        result.append({
            "speaker": segment["speaker"],
            "text": full_transcript  # TODO: chính xác hơn cần align timestamp
        })
    return result
```

### 3.3. Video Processing

```python
# video_processing.py
import cv2
import os

def extract_keyframes(video_path, output_dir, interval=30):
    """Extract 1 frame mỗi 30 frame"""
    cap = cv2.VideoCapture(video_path)
    count = 0
    frame_count = 0
    
    os.makedirs(output_dir, exist_ok=True)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if count % interval == 0:
            frame_path = f"{output_dir}/frame_{frame_count}.jpg"
            cv2.imwrite(frame_path, frame)
            frame_count += 1
        
        count += 1
    
    cap.release()
    return frame_count

# Extract audio từ video
from moviepy.editor import VideoFileClip

def extract_audio(video_path, output_audio="audio.mp3"):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(output_audio)
    return output_audio

# Pipeline đầy đủ
def process_video(video_path):
    # 1. Extract frames
    frames_dir = "video_frames"
    extract_keyframes(video_path, frames_dir)
    
    # 2. Extract + transcribe audio
    audio_path = extract_audio(video_path)
    transcript = transcribe_audio(audio_path)
    
    # 3. Caption cho từng frame
    captions = []
    for frame_file in os.listdir(frames_dir):
        caption = generate_caption(f"{frames_dir}/{frame_file}")
        captions.append(caption)
    
    return {
        "transcript": transcript,
        "frame_captions": captions
    }
```

---

## 4. Xây dựng Multi-modal Vector Database

### 4.1. Cài đặt Chroma (hỗ trợ multi-modal tốt)

```bash
pip install chromadb
pip install pillow
pip install pytesseract
pip install whisper
pip install opencv-python
pip install moviepy
```

### 4.2. Index nhiều loại media

```python
# multimodal_index.py
import chromadb
from chromadb.utils import embedding_functions
import base64
from io import BytesIO

client = chromadb.Client()

# Sử dụng CLIP embedding function (hỗ trợ cả text lẫn image)
clip_ef = embedding_functions.OpenCLIPEmbeddingFunction()

collection = client.create_collection(
    name="multimodal_docs",
    embedding_function=clip_ef,
    metadata={"description": "Multi-modal knowledge base"}
)

# Hàm helper
def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# 1. Index text documents
def index_text(text, metadata):
    collection.add(
        documents=[text],
        metadatas=[metadata],
        ids=[metadata["id"]]
    )

# 2. Index images
def index_image(image_path, metadata):
    # OCR extract text
    text = extract_text_from_image(image_path)
    
    # Caption
    caption = generate_caption(image_path)
    
    # Combine
    full_text = f"{caption}. {text}"
    
    # Store image as base64 in metadata
    metadata["image_base64"] = image_to_base64(image_path)
    metadata["type"] = "image"
    
    collection.add(
        documents=[full_text],
        metadatas=[metadata],
        ids=[metadata["id"]]
    )

# 3. Index audio
def index_audio(audio_path, metadata):
    transcript = transcribe_audio(audio_path)
    
    metadata["audio_path"] = audio_path
    metadata["type"] = "audio"
    
    collection.add(
        documents=[transcript],
        metadatas=[metadata],
        ids=[metadata["id"]]
    )

# 4. Index video
def index_video(video_path, metadata):
    video_data = process_video(video_path)
    
    # Combine all text
    full_text = video_data["transcript"] + " " + " ".join(video_data["frame_captions"])
    
    metadata["video_path"] = video_path
    metadata["type"] = "video"
    
    collection.add(
        documents=[full_text],
        metadatas=[metadata],
        ids=[metadata["id"]]
    )
```

### 4.3. Build index từ thư mục

```python
# build_multimodal_index.py
import os
from pathlib import Path

def build_index_from_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = Path(file).suffix.lower()
            
            metadata = {
                "id": file_path,
                "filename": file,
                "source": file_path
            }
            
            if file_ext == ".pdf":
                # Xử lý PDF (text extraction)
                from pypdf import PdfReader
                reader = PdfReader(file_path)
                text = "".join([page.extract_text() for page in reader.pages])
                index_text(text, metadata)
            
            elif file_ext in [".png", ".jpg", ".jpeg"]:
                index_image(file_path, metadata)
            
            elif file_ext in [".mp3", ".wav", ".m4a"]:
                index_audio(file_path, metadata)
            
            elif file_ext in [".mp4", ".avi", ".mov"]:
                index_video(file_path, metadata)
            
            print(f"Indexed: {file}")

# Chạy
build_index_from_folder("./knowledge_base")
```

---

## 5. Multi-modal Retrieval

### 5.1. Query với filter theo media type

```python
# multimodal_retrieval.py

def retrieve_multimodal(query, media_types=None, k=5):
    """
    query: câu hỏi của user
    media_types: ["text", "image", "audio", "video"] hoặc None (all)
    """
    
    # Build filter
    where_filter = None
    if media_types:
        where_filter = {"type": {"$in": media_types}}
    
    results = collection.query(
        query_texts=[query],
        n_results=k,
        where=where_filter
    )
    
    return results

# Ví dụ: chỉ tìm trong images
results = retrieve_multimodal("sales chart 2023", media_types=["image"])

# Ví dụ: tìm trong audio + video
results = retrieve_multimodal("what did John say about budget", media_types=["audio", "video"])
```

### 5.2. Kết hợp Image Search với Text Query

```python
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import numpy as np

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def search_images_by_text(query_text, image_paths, top_k=3):
    """Tìm ảnh relevant nhất dựa trên text query"""
    
    # Embed text query
    text_inputs = clip_processor(text=[query_text], return_tensors="pt")
    with torch.no_grad():
        text_embedding = clip_model.get_text_features(**text_inputs)
    
    # Embed tất cả images
    image_embeddings = []
    for img_path in image_paths:
        img = Image.open(img_path)
        img_inputs = clip_processor(images=img, return_tensors="pt")
        with torch.no_grad():
            img_emb = clip_model.get_image_features(**img_inputs)
        image_embeddings.append(img_emb)
    
    # Compute cosine similarity
    image_embeddings = torch.cat(image_embeddings)
    similarities = torch.nn.functional.cosine_similarity(
        text_embedding, image_embeddings
    )
    
    # Top-K
    top_indices = similarities.argsort(descending=True)[:top_k]
    
    results = [(image_paths[i], similarities[i].item()) for i in top_indices]
    return results

# Ví dụ
images = ["chart1.png", "photo1.jpg", "diagram1.png"]
results = search_images_by_text("revenue growth chart", images)
# Output: [("chart1.png", 0.92), ("diagram1.png", 0.78), ...]
```

---

## 6. Multi-modal Generation với Gemini

### 6.1. Gửi nhiều loại media cho LLM

```python
# multimodal_generation.py
from google import genai
import base64

client = genai.Client(api_key="YOUR_API_KEY")

def ask_multimodal_rag(query):
    # 1. Retrieve multi-modal context
    results = retrieve_multimodal(query, k=3)
    
    # 2. Prepare content cho Gemini
    contents = []
    
    for i, metadata in enumerate(results["metadatas"][0]):
        doc_type = metadata.get("type", "text")
        
        if doc_type == "text":
            contents.append({
                "type": "text",
                "text": results["documents"][0][i]
            })
        
        elif doc_type == "image":
            # Decode base64 image
            img_data = base64.b64decode(metadata["image_base64"])
            contents.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": metadata["image_base64"]
                }
            })
            contents.append({
                "type": "text",
                "text": results["documents"][0][i]  # Caption/OCR text
            })
        
        elif doc_type == "audio":
            # Audio → transcript đã được index
            contents.append({
                "type": "text",
                "text": f"[AUDIO TRANSCRIPT]: {results['documents'][0][i]}"
            })
    
    # 3. Add user query
    contents.append({
        "type": "text",
        "text": f"\n\nQuestion: {query}\n\nAnswer:"
    })
    
    # 4. Call Gemini
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=contents
    )
    
    return response.text

# Test
answer = ask_multimodal_rag("Show me the sales chart for Q4 and explain the trend")
print(answer)
```

### 6.2. Xử lý Image + Text query

```python
def ask_with_image(query_text, image_path):
    """User upload ảnh + hỏi câu hỏi"""
    
    # 1. Retrieve similar images from vector DB
    similar_results = retrieve_multimodal(query_text, media_types=["image"], k=2)
    
    # 2. Prepare prompt
    with open(image_path, "rb") as f:
        user_image_b64 = base64.b64encode(f.read()).decode()
    
    contents = [
        {
            "type": "text",
            "text": "Here are similar examples from our knowledge base:"
        }
    ]
    
    # Add retrieved images
    for metadata in similar_results["metadatas"][0]:
        contents.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": metadata["image_base64"]
            }
        })
    
    # Add user's image
    contents.append({
        "type": "text",
        "text": "Now, here is the user's image:"
    })
    contents.append({
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": user_image_b64
        }
    })
    
    # Add question
    contents.append({
        "type": "text",
        "text": f"\nQuestion: {query_text}"
    })
    
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=contents
    )
    
    return response.text
```

---

## 7. Advanced: ImageBind (Unified Embedding cho 7 modalities)

ImageBind của Meta hỗ trợ: text, image, audio, video, thermal, depth, IMU.

```python
# imagebind_example.py
# (Requires: pip install imagebind)

from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = imagebind_model.imagebind_huge(pretrained=True).to(device)
model.eval()

# Load data
inputs = {
    ModalityType.TEXT: data.load_and_transform_text(["a dog playing in park"], device),
    ModalityType.VISION: data.load_and_transform_vision_data(["dog.jpg"], device),
    ModalityType.AUDIO: data.load_and_transform_audio_data(["dog_bark.wav"], device),
}

# Generate embeddings
with torch.no_grad():
    embeddings = model(inputs)

# All modalities → same embedding space
text_emb = embeddings[ModalityType.TEXT]
image_emb = embeddings[ModalityType.VISION]
audio_emb = embeddings[ModalityType.AUDIO]

# Similarity
from torch.nn.functional import cosine_similarity
print(cosine_similarity(text_emb, image_emb))  # High if relevant
print(cosine_similarity(text_emb, audio_emb))
```

---

## 8. Use Cases thực tế

### 8.1. E-learning Platform

```
Student query: "Giải thích về cấu trúc tế bào thực vật"

Multi-modal retrieval:
- Text: Định nghĩa từ textbook
- Image: Diagram tế bào có nhãn
- Video: Clip giải thích 2 phút
- Audio: Bài giảng của giáo viên

LLM synthesis:
→ Tổng hợp text explanation + embed diagram + link video
```

### 8.2. Customer Support

```
User: Upload ảnh lỗi màn hình + "Laptop bị lỗi này, sửa thế nào?"

System:
1. CLIP tìm ảnh lỗi tương tự trong knowledge base
2. Retrieve hướng dẫn sửa lỗi (text + video tutorial)
3. Gemini Vision phân tích ảnh user → chẩn đoán
4. Trả về: Nguyên nhân + Hướng dẫn fix (text + video)
```

### 8.3. Meeting Intelligence

```
Input: Ghi âm meeting 1 giờ

Pipeline:
1. Whisper transcribe → text
2. Diarization → phân speaker
3. Extract action items
4. Index vào vector DB

Query: "What did Sarah say about the marketing budget?"
→ Retrieve đoạn audio timestamp + transcript
```

---

## 9. Đánh giá Multi-modal RAG

### 9.1. Metrics

- **Text Retrieval**: Recall@K, MRR (giống RAG thường)
- **Image Retrieval**: mAP (mean Average Precision), CLIP score
- **Audio Quality**: WER (Word Error Rate) của transcript
- **End-to-end**: User satisfaction, answer accuracy

### 9.2. Human Evaluation

Setup test set:
- 50 câu hỏi yêu cầu text
- 50 câu hỏi yêu cầu image
- 50 câu hỏi yêu cầu audio/video

Đánh giá:
1. Retrieval đúng không?
2. Media type phù hợp không?
3. Câu trả lời chính xác không?

---

## 10. Deployment Tips

### 10.1. Storage

- **Text**: Lưu trong vector DB (nhẹ)
- **Images**: Lưu file system, chỉ lưu path + base64 thumbnail trong DB
- **Audio/Video**: Cloud storage (S3, GCS), lưu URL trong DB

### 10.2. Performance

- **Lazy loading**: Chỉ load media khi cần (không load hết 100 images lúc retrieve)
- **Caching**: Cache embedding của media thường dùng
- **Batch processing**: Index nhiều file cùng lúc

### 10.3. Cost Optimization

| Media Type | Embedding Cost | Storage Cost | Retrieval Cost |
|------------|---------------|--------------|----------------|
| Text       | Thấp          | Thấp         | Thấp           |
| Image      | Trung bình    | Trung bình   | Trung bình     |
| Audio      | Cao (Whisper) | Cao          | Thấp (text)    |
| Video      | Rất cao       | Rất cao      | Trung bình     |

**Strategy**: 
- Transcribe audio/video offline (1 lần)
- Lưu transcript trong vector DB
- Chỉ serve media khi user request

---

## 11. Bài tập thực hành

### Bài 1: Image RAG
1. Thu thập 100 ảnh biểu đồ/diagram
2. OCR + Caption
3. Index vào Chroma với CLIP
4. Build search engine: text query → trả về ảnh relevant

### Bài 2: Audio RAG
1. Ghi âm 5 đoạn giải thích về topic khác nhau
2. Whisper transcribe
3. Index transcript
4. Query: "Giải thích về X" → trả về đoạn audio + timestamp

### Bài 3: Full Multi-modal RAG
1. Tạo knowledge base gồm: 10 PDF + 20 images + 5 audio
2. Index tất cả vào Chroma
3. Build chatbot hỗ trợ:
   - Text query → text/image/audio answer
   - Image upload → tìm ảnh tương tự + explain
   - Audio query → search trong audio knowledge base

---

## 12. Resources

### Models
- **CLIP**: https://github.com/openai/CLIP
- **Whisper**: https://github.com/openai/whisper
- **ImageBind**: https://github.com/facebookresearch/ImageBind
- **BLIP**: https://github.com/salesforce/BLIP

### Vector Databases
- **Chroma**: Multi-modal metadata support
- **Qdrant**: Native multi-modal filtering
- **Weaviate**: Multi-modal modules

### APIs
- **Gemini 2.0**: Native multi-modal input
- **GPT-4 Vision**: Image understanding
- **Claude 3**: Image + PDF support

---

## Kết luận

Multi-modal RAG mở rộng khả năng của hệ thống từ text sang image, audio, video. Điều này quan trọng trong các ứng dụng thực tế như e-learning, customer support, meeting intelligence.

**Key takeaways**:
- Sử dụng specialized models cho từng modality (Whisper, CLIP, OCR)
- Chuyển tất cả về text space hoặc unified embedding space
- Multi-modal LLM (Gemini, GPT-4V) để synthesis

**Next step**: Agent-based systems để orchestrate multi-modal workflow phức tạp.

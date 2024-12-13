# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import shutil
import zipfile
import base64
import csv
import time
from io import BytesIO
import cv2
import google.generativeai as genai
from cog import BasePredictor, Input, Path, Secret
from openai import OpenAI, OpenAIError
from anthropic import Anthropic
from PIL import Image

SUPPORTED_VIDEO_TYPES = (".mov", ".mp4")

class Predictor(BasePredictor):
    def setup(self):
        pass

    def predict(
        self,
        video_zip_archive: Path = Input(
            description="ZIP archive containing videos to process"
        ),
        include_csv: bool = Input(
            description="Whether to include CSV in output",
            default=True
        ),
        caption_prefix: str = Input(
            description="Optional prefix for video captions", default=""
        ),
        caption_suffix: str = Input(
            description="Optional suffix for video captions", default=""
        ),
        frames_to_extract: int = Input(
            description="Number of frames to extract from each video for analysis",
            default=2
        ),
        system_prompt: str = Input(
            description="System prompt for caption generation",
            default="""
            Analyze these frames from a video and write a detailed caption. 
            Describe the type of video (e.g., animation, live-action footage, etc.).
            Focus on consistent elements across frames and any notable motion or action.
            Describe the main subjects, setting, and overall mood of the video.
            Use clear, descriptive language suitable for text-to-video generation.
            """
        ),
        model: str = Input(
            description="AI model to use for captioning",
            choices=[
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo",
                "claude-3-5-sonnet-20240620",
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307",
                "gemini-1.5-pro",
                "gemini-1.5-flash",
            ],
            default="gpt-4o",
        ),
        openai_api_key: Secret = Input(description="API key for OpenAI", default=None),
        anthropic_api_key: Secret = Input(description="API key for Anthropic", default=None),
        google_generativeai_api_key: Secret = Input(description="API key for Google Generative AI", default=None),
    ) -> Path:
        # Cleanup
        if os.path.exists("/tmp/outputs"):
            shutil.rmtree("/tmp/outputs")
        if os.path.exists("/tmp/frames"):
            shutil.rmtree("/tmp/frames")
        os.makedirs("/tmp/outputs")
        os.makedirs("/tmp/frames")

        # Initialize API client based on model
        client = self.initialize_client(model, openai_api_key, anthropic_api_key, google_generativeai_api_key)

        self.extract_videos_from_zip(video_zip_archive, SUPPORTED_VIDEO_TYPES)

        video_count = sum(1 for filename in os.listdir("/tmp/outputs") 
                         if filename.lower().endswith(SUPPORTED_VIDEO_TYPES))
        print(f"Number of videos to be captioned: {video_count}")
        print("===================================================")

        results = []
        errors = []
        csv_path = os.path.join("/tmp/outputs", "captions.csv")
        with open(csv_path, "w", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["caption", "video_file"])

            for filename in os.listdir("/tmp/outputs"):
                if filename.lower().endswith(SUPPORTED_VIDEO_TYPES):
                    print(f"Processing {filename}")
                    try:
                        # Extract frames from video
                        frames = self.extract_frames(
                            os.path.join("/tmp/outputs", filename),
                            frames_to_extract
                        )
                        
                        # Generate caption from frames
                        caption = self.generate_video_caption(
                            frames,
                            model,
                            client,
                            caption_prefix,
                            caption_suffix,
                            filename,
                            system_prompt
                        )
                        
                        print(f"Caption: {caption}")

                        # Save caption to text file
                        txt_filename = os.path.splitext(filename)[0] + ".txt"
                        txt_path = os.path.join("/tmp/outputs", txt_filename)
                        with open(txt_path, "w") as txt_file:
                            txt_file.write(caption)

                        csvwriter.writerow([caption, filename])
                        results.append({"filename": filename, "caption": caption})
                        
                    except Exception as e:
                        print(f"Error processing {filename}: {str(e)}")
                        errors.append({"filename": filename, "error": str(e)})
                    print("===================================================")

        # Create output zip with captions
        output_zip_path = "/tmp/video_captions.zip"
        with zipfile.ZipFile(output_zip_path, "w") as zipf:
            for root, dirs, files in os.walk("/tmp/outputs"):
                for file in files:
                    if file.endswith(".txt") or (file.endswith(".csv") and include_csv):
                        zipf.write(os.path.join(root, file), file)

        if errors:
            print("\nError Summary:")
            for error in errors:
                print(f"File: {error['filename']}, Error: {error['error']}")

        return Path(output_zip_path)

    def extract_frames(self, video_path: str, num_frames: int) -> list:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
        frames = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                frames.append(frame_pil)
        
        cap.release()
        return frames

    def generate_video_caption(
        self,
        frames: list,
        model: str,
        client,
        caption_prefix: str,
        caption_suffix: str,
        video_filename: str,
        system_prompt: str
    ) -> str:
        message_content = f"Please analyze these frames from the video '{video_filename}' and provide a caption."
        if caption_prefix or caption_suffix:
            message_content = self.prepare_message_content(message_content, caption_prefix, caption_suffix)

        # Convert frames to base64
        base64_frames = []
        for frame in frames:
            with frame as img:
                buffered = BytesIO()
                img.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                base64_frames.append(img_str)

        # Generate caption based on model type
        if model.startswith("gpt"):
            return self.generate_openai_caption(model, client, system_prompt, message_content, base64_frames)
        elif model.startswith("claude"):
            return self.generate_claude_caption(model, client, system_prompt, message_content, base64_frames)
        elif model.startswith("gemini"):
            return self.generate_gemini_caption(client, system_prompt, message_content, frames)

    def extract_videos_from_zip(self, video_zip_archive: Path, supported_video_types: tuple):
        with zipfile.ZipFile(video_zip_archive, "r") as zip_ref:
            for file in zip_ref.namelist():
                if (
                    file.lower().endswith(supported_video_types)
                    and not file.startswith("__MACOSX/")
                    and not os.path.basename(file).startswith("._")
                ):
                    filename = os.path.basename(file)
                    source = zip_ref.open(file)
                    target = open(os.path.join("/tmp/outputs", filename), "wb")
                    with source, target:
                        shutil.copyfileobj(source, target)

        print("Files extracted:")
        for root, dirs, files in os.walk("/tmp/outputs"):
            for f in files:
                print(f"{os.path.join(root, f)}")

    def initialize_client(self, model, openai_key, anthropic_key, google_key):
        if model.startswith("gpt"):
            if not openai_key:
                raise ValueError("OpenAI API key is required for GPT models")
            return OpenAI(api_key=openai_key.get_secret_value())
        elif model.startswith("claude"):
            if not anthropic_key:
                raise ValueError("Anthropic API key is required for Claude models")
            return Anthropic(api_key=anthropic_key.get_secret_value())
        elif model.startswith("gemini"):
            if not google_key:
                raise ValueError("Google Generative AI API key is required for Gemini models")
            genai.configure(api_key=google_key.get_secret_value())
            return genai.GenerativeModel(model_name=model)

    # Existing helper methods would remain similar, modified to handle multiple frames
    def prepare_message_content(self, message_prompt: str, prefix: str, suffix: str) -> str:
        message_content = message_prompt
        if prefix and suffix:
            message_content += f"\n\nPlease prefix the caption with '{prefix}' and suffix it with '{suffix}'."
        elif prefix:
            message_content += f"\n\nPlease prefix the caption with '{prefix}'."
        elif suffix:
            message_content += f"\n\nPlease suffix the caption with '{suffix}'."
        return message_content

    def generate_openai_caption(self, model: str, client, system_prompt: str, message_content: str, base64_frames: list) -> str:
        messages = [{"role": "system", "content": system_prompt}]
        for idx, frame in enumerate(base64_frames):
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Frame {idx + 1}:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}}
                ]
            })
        
        messages.append({"role": "user", "content": message_content})
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=300
        )
        return response.choices[0].message.content

    def generate_claude_caption(self, model: str, client, system_prompt: str, message_content: str, base64_frames: list) -> str:
        content = []
        for idx, frame in enumerate(base64_frames):
            content.extend([
                {"type": "text", "text": f"Frame {idx + 1}:"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": frame}}
            ])
        
        content.append({"type": "text", "text": message_content})
        
        response = client.messages.create(
            model=model,
            max_tokens=300,
            system=system_prompt,
            messages=[{"role": "user", "content": content}]
        )
        return response.content[0].text

    def generate_gemini_caption(self, client, system_prompt: str, message_content: str, frames: list) -> str:
        # Gemini processes frames as a batch
        prompt = f"{system_prompt}\n\n{message_content}"
        response = client.generate_content([prompt] + frames)
        return response.text
# cog-batch-video-captioning

A cog model for batch image captioning using various AI from OpenAI, Anthropic, and Google's Generative AI:

https://replicate.com/lucataco/bulk-video-caption



## Features

- Process multiple images from a ZIP archive
- supports mov, mp4
- Customizable caption prefixes and suffixes
- Support for multiple AI models:
	- OpenAI: GPT-4 and variants
	- Anthropic: Claude-3.5, Claude-3 variants
	- Google: Gemini-1.5 variants
- Flexible system prompts
- Error handling and retry mechanism
- Output as a ZIP file containing captions that match image filenames as well as an optional CSV summary
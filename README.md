# CS5720 - Neural Networks and Deep Learning  
### Bonus Assignment – Summer 2025  
**Student Name:** Swapnil Mergu
**Student Id:** 700772464
**University of Central Missouri**  
**Course:** CS5720 Neural Networks and Deep Learning  

---
## Assignment Overview
This assignment provides an overview and implementation of Conditional GAN for label-controlled MNIST generation, as well as a Hugging Face QA pipeline for context-based answer extraction.

---
## 1. Transformers
Use Hugging Face’s transformers library to build a simple question answering system using pre-trained models.
Setup Instructions:
Before starting, make sure your Python environment has the transformers and torch libraries installed.

Assignment Tasks:

#### 1. Basic Pipeline Setup
- Import the pipeline function from transformers.
- Initialize a question-answering pipeline using the default model.
- Ask a question based on the given context.

Expected output:
- 'answer': 'Charles Babbage' (or close variant)
- A confidence 'score' key with a float value above 0.65
- Valid 'start' and 'end' indices

Output
<pre>
Basic Pipeline Result:
 {'score': 0.9978699684143066, 'start': 0, 'end': 15, 'answer': 'Charles Babbage'}
</pre>

#### 2. Use a Custom Pretrained Model
•	Switch to a different QA model like deepset/roberta-base-squad2.

Expected output:
- 'answer': 'Charles Babbage'
- 'score' greater than 0.70
- Include 'start' and 'end' indices

Output
<pre>
Custom Model Result:
 {'score': 0.9640105962753296, 'start': 0, 'end': 15, 'answer': 'Charles Babbage'}
</pre>

#### 3. Test on Your Own Example:
- Write your own 2–3 sentence context.
- Ask two different questions from it and print the answers.

Expected output:
- Include a relevant, meaningful 'answer' to each question
- Display a 'score' above 0.70 for each answer

Output:
<pre>
Your Own Example Results:
Q1: Who was the main lead in Pokiri movie?
A1: {'score': 0.9832262396812439, 'start': 27, 'end': 38, 'answer': 'Mahesh Babu'}
Q2: When was the Pokiri movie released?
A2: {'score': 0.9479328393936157, 'start': 72, 'end': 76, 'answer': '2006'}
</pre>

---
## 2. GAN
#### 1. Digit-Class Controlled Image Generation with Conditional GAN

Objective: 
Implement a Conditional GAN that generates MNIST digits based on a given class label (0–9). The goal is to understand how conditioning GANs on labels affects generation and how class control is added.

Task Description:
- Modify a basic GAN to accept a digit label as input.
- Concatenate the label embedding with both: the noise vector (input to Generator) and the image input (to the Discriminator).
- Train the cGAN on MNIST and generate digits conditioned on specific labels (e.g., generate only 3s or 7s).
- Visualize generated digits label by label (e.g., one row per digit class).

Expected Output:
- A row of 10 generated digits, each conditioned on labels 0 through 9.
- Generator should learn to control output based on class.
- Loss curves may still fluctuate, but quality and label accuracy improves over time.

Output:

![download](https://github.com/user-attachments/assets/11a84abd-5b8c-4a66-ba97-01fa4be0b6fa)
![download](https://github.com/user-attachments/assets/119dc8bc-9100-46fd-9083-9decb528f31b)

## Short Question Answers:
#### 1.	How does a Conditional GAN differ from a vanilla GAN? Include at least one real-world application where conditioning is important.

#### Answer:
A Conditional GAN (cGAN) differs from a vanilla GAN by taking both noise and a label as input, enabling it to generate data conditioned on specific classes. A real-world example is generating images of specific clothing types in fashion design based on category labels.

#### 2.	•	What does the discriminator learn in an image-to-image GAN? Why is pairing important in this context?

#### Answer:
In an image-to-image GAN, the discriminator learns to distinguish between real and generated image pairs. Pairing is important because it ensures the generator produces outputs that accurately correspond to the input image, preserving structure and content.

---
# How to Run

```bash
# 1. Clone the Repository
git clone <Transformers-GAN>
open Transformers-GAN/n-n-bonus-assignment.ipynb

# 2. Run the each Python Scripts cell

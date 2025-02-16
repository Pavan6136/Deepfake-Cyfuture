## Solution

The Deepfake Detection Tool is designed to classify images as either real or fake using deep learning. It leverages a Convolutional Neural Network (CNN) for image classification and integrates a user-friendly interface using Streamlit. Additionally, an API is implemented with FastAPI to handle predictions. The tool also incorporates GPT-2 from Hugging Face’s `transformers` library to generate AI-based explanations.

## Approach

1. **Dataset Preparation**  
   - Used the Kaggle Deepfake "Image Detection dataset".  
   - Organized into `train`, `val`, and `test` directories.

2. **Preprocessing & Augmentation**  
   - Used `ImageDataGenerator` for rescaling images and data augmentation.  
   - Resized images to `(150, 150)` and normalized pixel values.

3. **Model Development**  
   - Built a CNN with multiple convolutional and pooling layers.  
   - Used ReLU activation and dropout for better generalization.  
   - Compiled with `adam` optimizer and `binary_crossentropy` loss function.

4. **Model Training & Evaluation**  
   - Trained on augmented images for 10 epochs.  
   - Achieved ~78% accuracy on test data.

5. **Model Deployment**  
   - Saved the trained model as `deepfake_model.keras`.  
   - Implemented a FastAPI-based backend for prediction handling.  
   - Integrated Streamlit UI for easy image upload and visualization.

6. **AI-Powered Explanation**  
   - Used GPT-2 from Hugging Face to generate explanations for predictions.  
   - Provided real-time feedback on deepfake detection.

7. **Execution Commands**  
   - **Run UI**: `streamlit run app.py`  
   - **Run API**: `uvicorn api:app --host 0.0.0.0 --port 8080 --reload`
8. **User manual**
   - **Upload the image in ui you will see the result and gpt-generated text too
   - **You will see the last_prection in 0.0.0.0:8080/last_prediction as fake/real too.


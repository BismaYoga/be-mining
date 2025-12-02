import pickle
import os

MODEL_FILE = 'model.pkl'

if not os.path.exists(MODEL_FILE):
    print(f"ERROR: File '{MODEL_FILE}' tidak ditemukan di direktori ini.")
    exit()

try:
    with open(MODEL_FILE, 'rb') as file:
        loaded_model = pickle.load(file)
    print(f"SUCCESS: Model '{MODEL_FILE}' berhasil dimuat!")
except Exception as e:
    print(f"ERROR saat memuat model: {e}")
    exit()

truck_count = 4         
excavator_count = 2   
operator_count = 12      
weather = 1          

input_data = [truck_count, excavator_count, operator_count, weather]

prediction = loaded_model.predict([input_data]) 

print("\n--- Hasil Prediksi ---")
print(f"Input Fitur: {input_data}")
print(f"Hasil Prediksi Model: {prediction[0]:.2f}") 
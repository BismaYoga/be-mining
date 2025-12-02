import asyncio
import os 
import pickle 
import re 
from dotenv import load_dotenv 
from typing import List, Dict, Union, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService 
from google.genai import types 

app = FastAPI(
    title="Mining Prediction and Recommendation API",
    description="API untuk prediksi target produksi tambang menggunakan model ML dan Gemini Agent dengan dukungan sesi berlanjut."
)

load_dotenv(dotenv_path=".env") 

if not os.getenv("GEMINI_API_KEY"):
    print("FATAL: GEMINI_API_KEY tidak ditemukan. Agen tidak dapat diinisialisasi.")

MODEL_FILE = 'model.pkl'
loaded_model = None

try:
    with open(MODEL_FILE, 'rb') as file:
        loaded_model = pickle.load(file)
    print(f"SUCCESS: Model '{MODEL_FILE}' berhasil dimuat.")
except FileNotFoundError:
    print(f"ERROR: File '{MODEL_FILE}' tidak ditemukan. Model prediksi tidak akan berfungsi.")
except Exception as e:
    print(f"ERROR saat memuat model: {e}. Pastikan Scikit-learn terinstal.")

AGENT_NAME = "mining_prediction_agent"
APP_NAME = "agents" 
GEMINI_MODEL = "gemini-2.5-flash" 
session_service = InMemorySessionService() 
mining_agent: Optional[LlmAgent] = None


def predict_mining_target(scenarios: List[List[Union[int, float]]]) -> str:
    """Tool yang dipanggil oleh Gemini Agent untuk mendapatkan prediksi tonase."""
    if loaded_model is None:
        return "ERROR: Model prediksi belum dimuat atau gagal dimuat."
    
    if not scenarios:
        return "ERROR: Input skenario kosong."

    try:
        predictions = loaded_model.predict(scenarios)
        
        results = []
        for i, (input_data, target) in enumerate(zip(scenarios, predictions)):
            results.append({
                "id": i + 1,
                "trucks": input_data[0],
                "excavators": input_data[1],
                "operators": input_data[2],
                "weather": input_data[3],
                "predicted_tonnage": round(target, 2)
            })

        return f"PREDICTION_RESULTS: {results}"
        
    except Exception as e:
        return f"ERROR saat menjalankan prediksi: {e}. Pastikan dimensi input model benar."


def initialize_agent():
    """Menginisialisasi LlmAgent secara global dengan instruksi parsing yang ketat."""
    global mining_agent
    
    agent_instruction = """
    Anda adalah **Ahli Optimasi Sumber Daya Tambang** (Mining Data Analyst).

    **Pemetaan Cuaca:** Light Rain=0, Cloudy=1, Sunny=2
    
    **ATURAN KONTEKS:**
    1. Jika query baru adalah MODIFIKASI (misal: "tambah 2 truk"), Anda HARUS mengambil Target Tonase (TT), Truk (T), Ekskavator (E), Operator (O), dan Cuaca (C) dari riwayat percakapan terakhir dan menerapkan modifikasi tersebut.
    2. Jika query baru memiliki SEMUA parameter, gunakan yang baru.

    **Tugas Utama (Untuk Setiap Respon):**
    1.  **Ekstrak/Tentukan** nilai: Target Tonase (TT), Truk (T), Ekskavator (E), Operator (O), dan Cuaca (C) dari query saat ini ATAU dari konteks yang dimodifikasi.
    2.  Buat Skenario 1 (Kontrol: T, E, O, C) dan 3 Skenario modifikasi (S2, S3, S4).
    3.  Panggil Tool `predict_mining_target` dengan 4 skenario.
    4.  **Parsing dan Struktur Output (WAJIB KETAT):**
        a. Hitung hasil prediksi dan selisih mutlak dari TT untuk keempat skenario.
        b. **OUTPUT PART 1 (Analisis Awal):** Sajikan analisis Skenario Kontrol secara kompleks serta alasannya.
        c. **OUTPUT PART 2 (Rekomendasi):** Sajikan 3 skenario terbaik (termasuk Kontrol jika itu yang terbaik) yang paling mendekati TT.
        d. Gunakan format ketat berikut, pastikan SEMUA field terisi dengan **nilai numerik murni** (tanpa unit atau simbol kecuali titik desimal).

    [Analisis singkat Skenario Kontrol]
    Target_Tonase_Ekstrak: [Nilai TT, harus berupa angka]
    Prediksi_Kontrol: [Hasil Prediksi Kontrol, harus berupa angka]
    Selisih_Kontrol: [Selisih Mutlak Kontrol, harus berupa angka]
    ---END_ANALYSIS---
    
    Rekomendasi 1: [Judul Deskriptif R1, misal: 'Mengurangi Truk']
    Truk: [Nilai T]
    Ekskavator: [Nilai E]
    Operator: [Nilai O]
    Cuaca: [Nilai C]
    Prediksi: [Hasil Prediksi]
    Selisih: [Selisih Mutlak]
    Alasan: [Alasan mengapa ini efektif, bandingkan dengan kontrol atau TT]
    ---START_RECOMMENDATION---
    
    Rekomendasi 2: [Judul Deskriptif R2]
    Truk: [Nilai T]
    Ekskavator: [Nilai E]
    Operator: [Nilai O]
    Cuaca: [Nilai C]
    Prediksi: [Hasil Prediksi]
    Selisih: [Selisih Mutlak]
    Alasan: [Alasan mengapa ini efektif, bandingkan dengan kontrol atau TT]
    ---START_RECOMMENDATION---
    
    Rekomendasi 3: [Judul Deskriptif R3]
    Truk: [Nilai T]
    Ekskavator: [Nilai E]
    Operator: [Nilai O]
    Cuaca: [Nilai C]
    Prediksi: [Hasil Prediksi]
    Selisih: [Selisih Mutlak]
    Alasan: [Alasan mengapa ini efektif, bandingkan dengan kontrol atau TT]
    """

    mining_agent = LlmAgent(
        name=AGENT_NAME,
        model=GEMINI_MODEL,
        tools=[predict_mining_target], 
        instruction=agent_instruction,
        description="Memberikan analisis hasil input pengguna dan 3 rekomendasi konfigurasi alat berat terbaik untuk mencapai target produksi harian.",
    )
    print("SUCCESS: Gemini Agent berhasil diinisialisasi.")
    
initialize_agent() 


class RecommendationDetail(BaseModel):
    title: str = Field(..., description="Judul deskriptif rekomendasi.")
    trucks: int
    excavators: int
    operators: int
    weather: Union[int, str]
    predicted_tonnage: float
    difference_from_target: float
    rationale: str = Field(..., description="Alasan kuat dan berbasis data mengapa skenario ini direkomendasikan.")

class MiningInput(BaseModel):
    """Schema untuk data yang dikirimkan: pesan teks bebas."""
    user_id: str = Field(..., example="user_api_001", description="ID unik pengguna untuk sesi berlanjut.")
    query: str = Field(..., example="Saya ingin 80 ton. Saat ini pakai 12 Truk, 3 Ekskavator, 18 Operator, cuaca Cloudy. Beri 3 rekomendasi!", description="Pesan teks bebas yang berisi Target dan parameter tambang.")

class ParsedRecommendationResponse(BaseModel):
    """Schema untuk output yang sudah dipisah-pisah. Semua kolom penting WAJIB diisi."""
    status: str = Field("success")
    target_tonnage: int = Field(..., description="Target tonase yang diekstrak.") 
    initial_analysis_text: str = Field(..., description="Analisis Agent mengenai Skenario Kontrol (Input Asli).")
    initial_prediction: float = Field(..., description="Hasil prediksi tonase dari input asli.") 
    initial_difference: float = Field(..., description="Selisih mutlak dari input asli ke target.") 
    recommendations: List[RecommendationDetail] = Field(..., description="Daftar 3 skenario rekomendasi terbaik.")



def parse_agent_response(text: str) -> Dict[str, Union[str, float, int, List[Dict]]]:
    """
    Mengurai respons teks Agent yang memiliki format ketat menjadi struktur data Python.
    Memastikan semua field numerik dan alasan terisi.
    """
    
    if '---END_ANALYSIS---' not in text:
        return {"error": "Format respons Agent tidak valid: Tag END_ANALYSIS tidak ditemukan."}

    analysis_part, recs_raw = text.split('---END_ANALYSIS---', 1)

    target_match = re.search(r'Target_Tonase_Ekstrak:\s*([\d\.]+)', analysis_part)
    pred_match = re.search(r'Prediksi_Kontrol:\s*([\d\.]+)', analysis_part)
    diff_match = re.search(r'Selisih_Kontrol:\s*([\d\.]+)', analysis_part)

    try:
        initial_prediction = float(pred_match.group(1)) if pred_match else 0.0
        target_tonnage = int(float(target_match.group(1))) if target_match else 0
        initial_difference = float(diff_match.group(1)) if diff_match else 0.0
    except Exception as e:
        return {"error": f"Gagal mengurai nilai numerik dari analisis awal. Error: {e}"}

    analysis_text = re.sub(r'(Target_Tonase_Ekstrak|Prediksi_Kontrol|Selisih_Kontrol):\s*[\d\.]+', '', analysis_part).strip()

    recs_list = recs_raw.split('---START_RECOMMENDATION---')
    recommendations_data = []

    for rec_block in recs_list:
        rec_block = rec_block.strip()
        if not rec_block:
            continue

        data = {}

        title_match = re.search(r'Rekomendasi \d: (.*)', rec_block)
        data['title'] = title_match.group(1).strip() if title_match else "Rekomendasi Tanpa Judul"

        params = re.findall(r'(Truk|Ekskavator|Operator|Cuaca|Prediksi|Selisih|Alasan):\s*([^\n]+)', rec_block)
        
        mapping = {
            'Truk': 'trucks', 'Ekskavator': 'excavators', 'Operator': 'operators', 
            'Cuaca': 'weather', 'Prediksi': 'predicted_tonnage', 
            'Selisih': 'difference_from_target', 'Alasan': 'rationale'
        }
        
        for key, value in params:
            python_key = mapping.get(key)
            if python_key:
                cleaned_value = value.replace('Ton', '').replace(',', '').strip()
                try:
                    if python_key in ['trucks', 'excavators', 'operators']:
                        data[python_key] = int(float(cleaned_value)) 
                    elif python_key in ['predicted_tonnage', 'difference_from_target']:
                        data[python_key] = float(cleaned_value)
                    else:
                        data[python_key] = cleaned_value
                except ValueError:
                    data[python_key] = cleaned_value
                    
        if 'rationale' not in data:
            data['rationale'] = "Alasan tidak tersedia (Gagal parsing)."

        if len(data) > 1:
            recommendations_data.append(data)

    if target_tonnage == 0 or initial_prediction == 0.0:
        return {"error": f"Parsing gagal mendapatkan Target Tonase atau Prediksi Kontrol."}
        
    return {
        "initial_analysis_text": analysis_text,
        "initial_prediction": initial_prediction,
        "target_tonnage": target_tonnage,
        "initial_difference": initial_difference,
        "recommendations": recommendations_data
    }


@app.post("/predict_and_recommend", response_model=ParsedRecommendationResponse)
async def predict_and_recommend(data: MiningInput):
    """
    Menerima pesan teks bebas, menjalankan Agent, dan mengembalikan data terstruktur. 
    Menggunakan user_id untuk sesi berlanjut.
    """
    global mining_agent

    if mining_agent is None or os.getenv("GEMINI_API_KEY") is None:
        raise HTTPException(
            status_code=503, 
            detail="Layanan Agent atau Kunci API tidak tersedia."
        )

    query = data.query
    user_id = data.user_id 
    session_id = f"session_{user_id}" 

    try:
        await session_service.create_session(app_name=APP_NAME, user_id=user_id, session_id=session_id)
    except Exception:
        pass

    runner = Runner(agent=mining_agent, app_name=APP_NAME, session_service=session_service)

    final_response_text = ""
    try:
        async for event in runner.run_async(
            user_id=user_id, 
            session_id=session_id, 
            new_message=types.Content(role="user", parts=[types.Part(text=query)])
        ):
            if event.is_final_response() and event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        final_response_text += part.text
                        break 
        
        if not final_response_text:
              raise Exception("Agent tidak menghasilkan respons akhir yang valid.")

        parsed_data = parse_agent_response(final_response_text)
        
        if "error" in parsed_data:
              raise Exception(f"Gagal mem-parsing output Agent: {parsed_data['error']}")

        return ParsedRecommendationResponse(
            status="success",
            target_tonnage=parsed_data['target_tonnage'],
            initial_analysis_text=parsed_data['initial_analysis_text'],
            initial_prediction=parsed_data['initial_prediction'],
            initial_difference=parsed_data['initial_difference'],
            recommendations=parsed_data['recommendations']
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Kesalahan pada layanan Agent: {str(e)}")


@app.delete("/end_session/{user_id}")
async def end_session(user_id: str):
    """Menghapus sesi Agent berdasarkan user_id, memaksa sesi baru di permintaan berikutnya."""
    session_id = f"session_{user_id}"
    try:
        await session_service.delete_session(app_name=APP_NAME, user_id=user_id, session_id=session_id)
        return {"status": "success", "message": f"Sesi untuk user_id '{user_id}' berhasil diakhiri."}
    except Exception:
        return {"status": "info", "message": f"Sesi untuk user_id '{user_id}' tidak ditemukan atau sudah berakhir."}
from fastapi.responses import FileResponse
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import uvicorn
from typing import Union

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"API": "Proyecto 1 - Etapa 2"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.post("/predict")
async def upload_file(file: UploadFile = File(...)):
    # Verificar si el archivo subido es un archivo CSV
    if not file.filename.endswith(".csv"):
        return {"error": "Solo se permiten archivos CSV"}

    try:
        # Leer el archivo CSV con Pandas
        df_predict = pd.read_csv(file.file, sep=';')
        df_predict = df_predict['Textos_espanol']

        filename_model = "assets/tfidf_model.joblib"
        filename_transform = "assets/tfidf_transform.pkl"

        # Carga el modelo desde el archivo
        tfidf_model = joblib.load(filename_model)
        tfidf = joblib.load(filename_transform)

        # Prediccion:
        tfidf.transform(df_predict)
        y_test_search_predict = tfidf_model.predict(df_predict)

        df_final = pd.DataFrame(df_predict)

        print(df_final)

        # Insertar la nueva columna en la posición deseada
        df_final['sdg'] = y_test_search_predict[0:]

        # Guardar DataFrame como un archivo CSV
        output_filename = "output_predictions.csv"
        df_final.to_csv(output_filename)

        # Devolver el archivo como respuesta
        return FileResponse(output_filename, filename="output_predictions.csv")
    except Exception as e:
        return {"error": e.__str__()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

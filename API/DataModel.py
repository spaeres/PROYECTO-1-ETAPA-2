from pydantic import BaseModel


class DataModel(BaseModel):
    # Estas varibles permiten que la librería pydantic haga el parseo entre el Json recibido y el modelo declarado.
    Textos_espanol: str

    # COMPLETAR CON LAS VARIABLES EXPLICATIVAS DE LOS DATOS DEL LABORATORIO 2

    # Esta función retorna los nombres de las columnas correspondientes con el modelo esxportado en joblib.
    def columns(self):
        return ["Textos_espanol"]

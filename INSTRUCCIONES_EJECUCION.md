# Instrucciones de ejecución

## Requisitos previos

- Python 3.10 o superior
- Entorno virtual activo (.venv)

## Instalación de dependencias

```bash
pip install -r requirements.txt
```

## Flujo de ejecución

### 1. Preparar los datos

```bash
python src/preparar_datos.py
```

### 2. Entrenar el modelo

```bash
python src/entrenar_modelo.py
```

### 3. Evaluar el modelo

```bash
python src/evaluar_modelo.py
```

### 4. Ejecutar la API

```bash
uvicorn api.main:app --reload
```

Acceder a: http://127.0.0.1:8000/docs

### 5. Ejecutar las pruebas

```bash
pytest
```

## Nota sobre trazabilidad

Cada cambio importante del proyecto debe registrarse mediante commits claros y descriptivos.

Esto permite conocer la evolución del código, la documentación, las pruebas y la configuración del proyecto.

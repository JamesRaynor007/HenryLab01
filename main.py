from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import pandas as pd
import os
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Define the paths for the CSV files
file_path_monthly = os.path.join(os.path.dirname(__file__), 'PeliculasPorMesListo.csv')
file_path_daily = os.path.join(os.path.dirname(__file__), 'PeliculasPorDiaListo.csv')
votes_file_path = os.path.join(os.path.dirname(__file__), 'FuncionVotos.csv')
scores_file_path = os.path.join(os.path.dirname(__file__), 'FuncionScore.csv')
resultado_crew_path = os.path.join(os.path.dirname(__file__), 'resultado_crew.csv')
funcion_director_path = os.path.join(os.path.dirname(__file__), 'FuncionDirector.csv')
resultado_cast_actores_path = os.path.join(os.path.dirname(__file__), 'ResultadoCastActores.csv')
funcion_actor_path = os.path.join(os.path.dirname(__file__), 'FuncionActor.csv')
lista_actores_path = os.path.join(os.path.dirname(__file__), 'ListaActores.csv')
file_path = os.path.join(os.path.dirname(__file__), 'data_reduction.csv')  # Cambia esto a la ruta de tu dataset


# Create a dictionary to map Spanish months to English months
meses_map = {
    'enero': 'January',
    'febrero': 'February',
    'marzo': 'March',
    'abril': 'April',
    'mayo': 'May',
    'junio': 'June',
    'julio': 'July',
    'agosto': 'August',
    'septiembre': 'September',
    'octubre': 'October',
    'noviembre': 'November',
    'diciembre': 'December'
}

# Create a dictionary to map Spanish days to English days
dias_map = {
    'lunes': 'Monday',
    'martes': 'Tuesday',
    'miercoles': 'Wednesday',
    'jueves': 'Thursday',
    'viernes': 'Friday',
    'sabado': 'Saturday',
    'domingo': 'Sunday',
}


class MessageResponse(BaseModel):
    mensaje: str  # Mensaje personalizado

class MovieInfo(BaseModel):
    title: str
    release_date: str
    return_: str  # Retorno como porcentaje
    budget: str   # Presupuesto formateado
    revenue: str  # Ingresos formateados

class DirectorResponse(BaseModel):
    resultado_texto: str  # Agregar el texto descriptivo aquí
    movies: List[MovieInfo]

# Load datasets
try:
    df_monthly = pd.read_csv(file_path_monthly)
    df_daily = pd.read_csv(file_path_daily)
    votes_df = pd.read_csv(votes_file_path)
    scores_df = pd.read_csv(scores_file_path)
    resultado_crew = pd.read_csv(resultado_crew_path)
    funcion_director = pd.read_csv(funcion_director_path)
    resultado_cast_actores = pd.read_csv(resultado_cast_actores_path)
    funcion_actor = pd.read_csv(funcion_actor_path)
    lista_actores = pd.read_csv(lista_actores_path)
    data = pd.read_csv(file_path)
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Error al cargar los archivos: {str(e)}")

# Inicializar el vectorizador y calcular la matriz TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(data['TokensLista'])

# Ensure required columns are present
for df, required_columns in [
    (df_monthly, ['title', 'month']),
    (df_daily, ['title', 'day_of_week']),
    (votes_df, ['title', 'vote_count', 'vote_average']),
    (scores_df, ['title', 'release_year', 'popularity']),
    (resultado_crew, ['movie_id', 'name']),
    (funcion_director, ['id', 'revenue', 'return']),
    (resultado_cast_actores, ['movie_id', 'name']),
    (funcion_actor, ['id', 'return']),
    (lista_actores, ['name']),
]:
    if not all(column in df.columns for column in required_columns):
        raise HTTPException(status_code=500, detail="El DataFrame no contiene las columnas esperadas.")

# Convert columns to lowercase
df_monthly.columns = df_monthly.columns.str.lower()
df_daily.columns = df_daily.columns.str.lower()
votes_df.columns = votes_df.columns.str.lower()
scores_df.columns = scores_df.columns.str.lower()
resultado_crew.columns = resultado_crew.columns.str.lower()
funcion_director.columns = funcion_director.columns.str.lower()
resultado_cast_actores.columns = resultado_cast_actores.columns.str.lower()
funcion_actor.columns = funcion_actor.columns.str.lower()
lista_actores.columns = lista_actores.columns.str.lower()

app = FastAPI(
    title="API de Películas",
    description="Esta API permite consultar información sobre películas, sus votaciones, puntuaciones, directores y actores.",
    version="1.0.0",
)


@app.get("/", response_model=dict)
async def read_root(request: Request):
    base_url = str(request.url).rstrip('/')
    return {
        "Mensaje": "Bienvenido a la API de películas.",
        "Instrucciones": (
            "Utiliza los siguientes endpoints para interactuar con la API:",
            "[cantidad de peliculas estrenadas en un mes (desde 1874 hasta 2020)] /peliculas/mes/?mes=nombre_del_mes",
            "[cantidad de peliculas estrenadas en un día (desde 1874 hasta 2020)] /peliculas/dia/?dia=nombre_del_dia",
            "[valoraciones de una pelicula y valoración promedio] /votes/?title=nombre_pelicula",
            "[popularidad y año de estreno de una pelicula] /score/?title=nombre_pelicula",
            "[titulos disponibles para consultar] /titles/",
            "[éxito de un director, ganancias de sus peliculas, retornos, promedios, cantidad de peliculas dirigidas y lista de ellas] /director/{director_name}",
            "[directores disponibles para consultar] /directores",
            "[actuaciones de un actor, retorno, promedio] /actor/{actor_name}",
            "[actores disponibles para consultar] /actores",
            "Para obtener recomendaciones, utiliza el endpoint'/recommendations/?title={tu_titulo}'",
            "Por ejemplo, para obtener recomendaciones para 'Inception', ",
            ),
        "Links Ejemplo": [
            {"Para Mes": list(meses_map.keys())[0], "url": f"{base_url}/peliculas/mes/?mes={list(meses_map.keys())[0]}"},
            {"Para Dia": list(dias_map.keys())[0], "url": f"{base_url}/peliculas/dia/?dia={list(dias_map.keys())[0]}"},
            {"Para Votación": f"{base_url}/votes/?title=Inception", "Descripción": "Buscar votación de una película"},
            {"Para Puntuación": f"{base_url}/score/?title=Toy%20Story", "Descripción": "Buscar puntuación de una película"},
            {"Para Títulos": f"{base_url}/titles/", "Descripción": "Listar todos los títulos"},
            {"Para Información del Director": f"{base_url}/director/Quentin%20Tarantino", "Descripción": "Obtener información de un director"},
            {"Para Todos los Directores": f"{base_url}/directores", "Descripción": "Listar todos los directores"},
            {"Para Información del Actor": f"{base_url}/actor/Leonardo%20DiCaprio", "Descripción": "Obtener información de un actor"},
            {"Para Todos los Actores": f"{base_url}/actores", "Descripción": "Listar todos los actores"},
            {"Para Recomendaciones": f"{base_url}/recommendations/?title=Inception"}
        ]
    }

@app.get("/peliculas/mes/", response_model=MessageResponse)
def get_peliculas_mes(mes: str):
    mes = mes.lower()
    if mes not in meses_map:
        raise HTTPException(status_code=400, detail="Mes no válido. Por favor ingrese un mes en español.")

    mes_en_ingles = meses_map[mes]
    resultado = df_monthly[df_monthly['month'] == mes_en_ingles]
    cantidad = resultado['title'].count() if not resultado.empty else 0

    return MessageResponse(
        mensaje=f"Cantidad de películas que fueron estrenadas en el mes de {mes_en_ingles}: {cantidad}"
    )

@app.get("/peliculas/dia/", response_model=MessageResponse)
def get_peliculas_dia(dia: str):
    dia = dia.lower()
    if dia not in dias_map:
        raise HTTPException(status_code=400, detail="Día no válido. Por favor ingrese un día en español.")

    dia_en_ingles = dias_map[dia]
    cantidad = df_daily[df_daily['day_of_week'] == dia_en_ingles].shape[0]

    return MessageResponse(
        mensaje=f"Cantidad de películas que fueron estrenadas en el día {dia_en_ingles}: {cantidad}"
    )

@app.get("/votes/")
async def get_movie_votes(title: str):
    movie = votes_df[votes_df['title'].str.lower() == title.lower()]
    if movie.empty:
        raise HTTPException(status_code=404, detail="Película no encontrada.")
    
    movie_data = movie.iloc[0]
    if movie_data['vote_count'] < 2000:
        return {
            "message": f"La película '{movie_data['title']}' tuvo menos de 2000 valoraciones."
        }
    else:
        return {
            "message": f"La película '{movie_data['title']}' tuvo {int(movie_data['vote_count'])} votos y su puntaje promedio fue {float(movie_data['vote_average']):.2f}."
        }

@app.get("/score/")
async def get_movie_score(title: str):
    movie = scores_df[scores_df['title'].str.lower() == title.lower()]
    if movie.empty:
        raise HTTPException(status_code=404, detail="Película no encontrada.")
    
    movie_data = movie.iloc[0]
    return {
        "message": f"La película '{movie_data['title']}' fue estrenada en el año {int(movie_data['release_year'])}, con una popularidad de {float(movie_data['popularity']):.2f}."
    }

@app.get("/titles/")
async def get_titles():
    return votes_df['title'].tolist()  # Asumiendo que ambos DataFrames tienen los mismos títulos

@app.get("/director/{director_name}", response_model=DirectorResponse)
def get_director_info(director_name: str):
    director_name_lower = director_name.lower()
    director_movies = resultado_crew[resultado_crew['name'].str.lower() == director_name_lower]

    if director_movies.empty:
        raise HTTPException(status_code=404, detail="Director no encontrado")

    director_movies = director_movies.merge(funcion_director, left_on='movie_id', right_on='id', how='inner')

    total_revenue = director_movies['revenue'].sum()
    total_return = director_movies['return'].sum()
    average_return = total_return / len(director_movies) if len(director_movies) > 0 else 0

    non_zero_returns = director_movies[director_movies['return'] > 0]
    average_return_non_zero = non_zero_returns['return'].mean() if len(non_zero_returns) > 0 else 0

    total_movies = len(director_movies)
    zero_return_movies = director_movies[director_movies['return'] == 0]
    total_zero_return = len(zero_return_movies)

    # Generar la lista de información de las películas
    movies_info = [
        MovieInfo(
            title=row['title'],
            release_date=row['release_date'],
            return_=f"{row['return']:.2f}%",
            budget=f"${row['budget']:,.2f}",
            revenue=f"${row['revenue']:,.2f}"
        ) for index, row in director_movies.iterrows()
    ]

    # Generar el texto descriptivo
    resultado_texto = (
        f"El director {director_name} ha obtenido una ganancia total de {total_revenue:,.2f}, "
        f"con un retorno total promedio de {average_return:.2f}% en un total de {total_movies} películas, "
        f"y con un retorno de {average_return_non_zero:.2f}% sin contar las "
        f"películas que no tienen retorno en este dataset.({total_zero_return})"
    )

    return DirectorResponse(
        resultado_texto=resultado_texto,
        movies=movies_info
    )

@app.get("/directores")
def obtener_directores():
    directores = resultado_crew['name'].unique().tolist()
    return directores

@app.get("/actor/{actor_name}")
def obtener_retorno_actor(actor_name: str):
    if not actor_name:
        raise HTTPException(status_code=400, detail="El nombre del actor no puede estar vacío.")
    
    actor_name_normalizado = actor_name.lower()
    peliculas_actor = resultado_cast_actores[resultado_cast_actores['name'].str.lower() == actor_name_normalizado]

    if peliculas_actor.empty:
        raise HTTPException(status_code=404, detail="Actor no encontrado")

    movie_ids = peliculas_actor['movie_id'].tolist()
    ganancias_actor = funcion_actor[funcion_actor['id'].isin(movie_ids)]

    retorno_total = ganancias_actor['return'].sum()
    ganancias_validas = ganancias_actor[ganancias_actor['return'] > 0]
    cantidad_peliculas_validas = len(ganancias_validas)

    if cantidad_peliculas_validas > 0:
        retorno_promedio = round(ganancias_validas['return'].mean(), 2) * 100
    else:
        retorno_promedio = 0.0

    retorno_total_formateado = f"{retorno_total * 100:,.2f}%"
    retorno_promedio_formateado = f"{retorno_promedio:,.2f}%"

    peliculas_con_return_zero = ganancias_actor[ganancias_actor['return'] == 0]['id'].tolist()
    peliculas_con_return_zero_count = len(peliculas_con_return_zero)

    # Generar el texto descriptivo
    resultado_texto = (
        f"El actor {actor_name} ha actuado en {len(ganancias_actor)} películas, "
        f"con un retorno total de {retorno_total_formateado}, "
        f"y un retorno promedio de {retorno_promedio_formateado}. "
        f"La cantidad de películas sin retorno en el dataset son {peliculas_con_return_zero_count}, "
        f"el retorno promedio contándolas es de {round(retorno_total / len(ganancias_actor) * 100, 2):,.2f}%."
    )

    return {"resultado": resultado_texto}

@app.get("/actores")
def listar_actores():
    actores_lista = lista_actores['name'].str.lower().tolist()
    return {"actores": actores_lista}


@app.get("/recommendations/")
def get_recommendations(title: str):
    titulo_ingresado = title.lower()  # Convertir a minúsculas

    # Convertir todos los títulos del dataset a minúsculas para la comparación
    data['lower_title'] = data['title'].str.lower()

    # Verificar si la película ingresada existe en el dataset
    if titulo_ingresado not in data['lower_title'].values:
        raise HTTPException(status_code=404, detail="Movie not found")

    # Obtener el índice de la película ingresada
    idx = data[data['lower_title'] == titulo_ingresado].index[0]

    # Calcular la similitud
    cosine_similarities = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    # Obtener las recomendaciones
    recommendations_indices = cosine_similarities.argsort()[-6:-1][::-1]  # 5 recomendaciones

    # Combina con el voto promedio
    recommendations = data.iloc[recommendations_indices]
    recommendations['similarity'] = cosine_similarities[recommendations_indices]
    recommendations = recommendations.sort_values(by='vote_average', ascending=False)

    # Retornar solo los campos deseados
    return recommendations[['title', 'vote_average', 'similarity']].to_dict(orient='records')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

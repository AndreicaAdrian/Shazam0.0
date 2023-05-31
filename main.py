import librosa
from scipy.spatial import distance

# Funcție pentru extragerea caracteristicilor audio
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, duration=10)  # Încărcăm fișierul audio și preluăm o secțiune de 10 secunde
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    return chroma_stft, mfcc

# Funcție pentru crearea bazei de date cu caracteristici audio
def create_database(audio_paths):
    database = {}
    for path in audio_paths:
        chroma_stft, mfcc = extract_features(path)
        database[path] = (chroma_stft, mfcc)
    return database

# Funcție pentru identificarea melodiei
def identify_song(query_path, database):
    query_chroma_stft, query_mfcc = extract_features(query_path)
    min_distance = float('inf')
    identified_song = None

    for path, (chroma_stft, mfcc) in database.items():
        distance_chroma = distance.euclidean(query_chroma_stft.flatten(), chroma_stft.flatten())
        distance_mfcc = distance.euclidean(query_mfcc.flatten(), mfcc.flatten())
        total_distance = distance_chroma + distance_mfcc

        if total_distance < min_distance:
            min_distance = total_distance
            identified_song = path

    return identified_song

# Calea către fișierele audio din baza de date
audio_paths = ["recording1.wav"]

# Crearea bazei de date
database = create_database(audio_paths)

# Calea către fișierul audio pe care dorim să-l identificăm
query_path = "recording1.wav"

# Identificarea melodiei
identified_song = identify_song(query_path, database)

# Afișarea melodiei identificate
print("Melodia identificată este:", identified_song)

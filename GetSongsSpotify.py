#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract songs features from Spotify API 


Created on Thu Jul 30 21:27:35 2020

@author: leonardo
"""

##Import Libs 
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials #To access authorised Spotify data
import time 
import numpy as np 
import pandas as pd

def albumSongs(uri): 

    album = uri #assign album uri to a_name 
    spotify_albums[album] = {} #Creates dictionary for that specific album 
    #Create keys-values of empty lists inside nested dictionary for album 
    spotify_albums[album]['album'] = [] #create empty list 
    spotify_albums[album]['track_number'] = [] 
    spotify_albums[album]['id'] = [] 
    spotify_albums[album]['name'] = [] 
    spotify_albums[album]['uri'] = [] 
    global tracks
    tracks = sp.album_tracks(album) #pull data on album tracks 
    
    
    for n in range(len(tracks['items'])): #for each song track 
        spotify_albums[album]['album'].append(album_names[album_count]) #append album name tracked via album_count 
        spotify_albums[album]['track_number'].append(tracks['items'][n]['track_number']) 
        spotify_albums[album]['id'].append(tracks['items'][n]['id']) 
        spotify_albums[album]['name'].append(tracks['items'][n]['name']) 
        spotify_albums[album]['uri'].append(tracks['items'][n]['uri'])

    spotify_albums[album]['acousticness'] = [] 
    spotify_albums[album]['danceability'] = [] 
    spotify_albums[album]['energy'] = [] 
    spotify_albums[album]['instrumentalness'] = [] 
    spotify_albums[album]['liveness'] = [] 
    spotify_albums[album]['loudness'] = [] 
    spotify_albums[album]['speechiness'] = [] 
    spotify_albums[album]['tempo'] = [] 
    spotify_albums[album]['valence'] = [] 
    spotify_albums[album]['popularity'] = [] 
    track_count = 0 
    for track in spotify_albums[album]['uri']: 
        global features
        #pull audio features per track  
        features = sp.audio_features(track) 

        #Append to relevant key-value 
        spotify_albums[album]['acousticness'].append(features[0]['acousticness']) 
        spotify_albums[album]['danceability'].append(features[0]['danceability']) 
        spotify_albums[album]['energy'].append(features[0]['energy']) 
        spotify_albums[album]['instrumentalness'].append(features[0]['instrumentalness']) 
        spotify_albums[album]['liveness'].append(features[0]['liveness']) 
        spotify_albums[album]['loudness'].append(features[0]['loudness']) 
        spotify_albums[album]['speechiness'].append(features[0]['speechiness']) 
        spotify_albums[album]['tempo'].append(features[0]['tempo']) 
        spotify_albums[album]['valence'].append(features[0]['valence']) 
        #popularity is stored elsewhere 
        pop = sp.track(track) 
        spotify_albums[album]['popularity'].append(pop['popularity']) 
        track_count+=1
        
def get_dict_df(spotify_albums_aux,artist):
    dic_df = {} 
    dic_df['album'] = [] 
    dic_df['track_number'] = [] 
    dic_df['id'] = [] 
    dic_df['name'] = [] 
    dic_df['uri'] = [] 
    dic_df['acousticness'] = [] 
    dic_df['danceability'] = [] 
    dic_df['energy'] = [] 
    dic_df['instrumentalness'] = [] 
    dic_df['liveness'] = [] 
    dic_df['loudness'] = [] 
    dic_df['speechiness'] = [] 
    dic_df['tempo'] = [] 
    dic_df['valence'] = [] 
    dic_df['popularity'] = [] 
    
    for i in spotify_albums_aux: 
        for j in spotify_albums_aux[i]: 
            dic_df[j].extend(spotify_albums_aux[i][j]) 
            len(dic_df['album'])
    
    df = pd.DataFrame.from_dict(dic_df) 
    df['artist'] = artist 
    return(df)

def get_songs_features(client_id,client_secret,artist):

    global album_count
    global spotify_albums
    global album_names
    global album_uris
    global sp
    
    client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret) 
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager) #spotify object to access API 

    result = sp.search(artist) #search query 

    #Extract Artist's uri 
    artist_uri = result['tracks']['items'][0]['artists'][0]['uri'] 
    sp_albums = sp.artist_albums(artist_uri, album_type='album') #Store artist's albums' names' and uris in separate lists 
    album_names = [i['name'] for i in sp_albums['items']]
    album_uris = [i['uri'] for i in sp_albums['items']]
    
    global album_count
    global spotify_albums
    
    spotify_albums = {} 
    album_count = 0 
    sleep_min = 2 
    sleep_max = 5 
    start_time = time.time() 
    request_count = 0 
    
    for i in album_uris: #each album 
        albumSongs(i) 
        print("Album " + str(album_names[album_count]) + " songs has been added to spotify_albums dictionary") 
        album_count+=1 #Updates album count once all tracks have been added
        request_count+=1 
        if request_count % 5 == 0: 
            print(str(request_count) + " playlists completed") 
            time.sleep(np.random.uniform(sleep_min, sleep_max)) 
            print('Loop #: {}'.format(request_count)) 
            print('Elapsed Time: {} seconds'.format(time.time() - start_time))
    dados=get_dict_df(spotify_albums,artist)        
    return(dados)

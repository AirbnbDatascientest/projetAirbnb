# -*- coding: utf-8 -*-
"""test_prediction_superhost.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1PB4O65WO_tJEZAO_LTH-VzW8ThWUUWQ8

# **Importation des Librairies**
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from google.colab import drive
drive.mount("/content/gdrive")

"""# **Ouverture du csv, un peu long car volumineux**"""

df=pd.read_csv('/content/gdrive/My Drive/Val_projet/airbnb.csv',";")

"""Le dataframe est immense et nous n'allons pas tout utiliser, nous supprimons toutes les colonnes qui ne sont pas utiles à notre recherche. Nous pourrons toujours venir les rechercher si nous en avons le besoin

"""

#Premiere supression de variables pas utiles à notre analyse
df=df.drop(['Listing Url','Scrape ID','Last Scraped','Host Acceptance Rate','Scrape ID','Last Scraped',
            'Thumbnail Url','Medium Url','Picture Url','XL Picture Url','Host URL','Host Acceptance Rate',
           'Host Thumbnail Url','Host Picture Url','Host Verifications','Neighbourhood','Neighbourhood Group Cleansed',
           'Geolocation','Has Availability','Calendar last Scraped','License','Jurisdiction Names'], axis=1)

#deuxieme supression de variables pas utiles à notre analyse
df=df.drop(['Summary','Space','Experiences Offered','Notes','Access','State','Zipcode','Host Name','Host Location','Host Neighbourhood', 'Host ID',
           'Host Listings Count','Smart Location','Country Code','Square Feet','Security Deposit','Calendar Updated',
            'Guests Included','Extra People','Street','Country','Cancellation Policy','Calculated host listings count','Features','Market'], axis=1)

#On rename les colonnes qui ont des espaces
df.rename(columns={'ID':"id",'Name': 'name', 'Description': 'description', 'Host About': 'host_about', 'Host Response Time': 'host_response_time', 'Host Response Rate': 'host_response_rate', 'Host Total Listings Count': 'host_total_listing_count', 'Neighbourhood Cleansed': 'neighbourhood_cleansed' , 'City': 'city', 'Property Type': 'property_type', 'Room Type': 'room_type', 'Accommodates': 'accommodates'
, 'Bathrooms': 'bathrooms', 'Bedrooms': 'bedrooms', 'Beds': 'beds', 'Bed Type': 'bed_type', 'Amenities': 'amenities', 'Price': 'price', 'Weekly Price': 'weekly_price', 'Monthly Price': 'monthly_price', 'Cleaning Fee': 'cleaning_fee', 'Minimum Nights': 'minimum_nights', 'Maximum Nights': 'maximum_nights', 'Availability 30': 'availability_30', 'Availability 60': 'availability_60', 'Availability 90': 'availability_90', 'Availability 365': 'availability_365', 'Number of Reviews': 'number_of_reviews', 'First Review': 'first_review'
, 'Last Review': 'last_review', 'Review Scores Rating': 'review_scores_rating', 'Review Scores Accuracy': 'review_scores_accuracy', 'Review Scores Cleanliness': 'review_scores_cleanliness', 'Review Scores Checkin': 'review_scores_checkin', 'Review Scores Communication': 'review_scores_communication', 'Review Scores Location': 'review_scores_location', 'Review Scores Value': 'review_scores_value', 'Reviews per Month': 'reviews_per_month', 'Longitude': 'longitude','Latitude': 'latitude',"Neighborhood Overview":"neighborhood_overview","Transit":"transit","Interaction":"interaction","House Rules":"house_rules","Host Since":"host_since"}, inplace=True)

# on met la colonne id en index

df=df.set_index("id")

#Nous allons nous concentrer sur les villes de Paris et Londres, avant de commencer le nettoyage nous allons garder uniquement toutes les données concernant ces deux villes

#on divise le df par villes puis on les regroupe dans df
df_paris=df[df.city=='Paris']
df_london=df[df.city=='London']
df = df_paris.append(df_london)

#On verifie qu'il reste bien uniquement Paris et Londre
print(df['city'].unique())

"""# **Analyse du dataframe**"""

#informations
df.info()

#Somme des Nans par colonne
df.isna().sum()

#Somme des Nans par colonne
print('\n Données manquantes par colonne :\n')
print(df.isna().sum())

#Verification des doublons
df.duplicated().sum()

#Nombre de valeur unique par colonne
df.nunique()

#description
df.describe()

"""# **Démarrons maintenant le nettoyage**"""

#On supprime les lignes dont les reviews ne sont pas renseignés étant donnée que c'est un point important de notre analyse
df=df.dropna(axis=0,subset=["first_review","last_review","review_scores_rating","review_scores_accuracy","review_scores_cleanliness","review_scores_checkin","review_scores_communication","review_scores_location","review_scores_value","reviews_per_month"])

#Maintenant que toutes les données ont une note attribuée, nous allons affiner en retirant les lignes qui ont des nans et qui peuvent fausser notre analyse.
df=df.dropna(axis=0,subset=["description","name","host_since","host_total_listing_count","price"])

#On supprime ces deux variables qui présentent trop de données manquantes pour être exploitées et qui sont difficiles à remplacer sans impacter les résultats de l'analyse
#df=df.drop(['host_response_time','host_response_rate'], axis=1)

# on remplace les nans des colonnes non renseigné volontairement par "nr", et "0" pour les colonnes numériques 'weekly_price','montly_price' et 'cleaning_fee'
# De plus, on remplace par 0 les Nans de 'bathrooms','bedrooms' et 'bed' que l'on conciderera par non renseigné par la suite

# afin de tout de même prendre en compte cette donnée et de ne pas fausser les analyses futures
# En effet, il y a énormément d'annonces dont ces colonnes ne sont pas remplies donc on ne peut pas simplement les supprimer

#je souhaitais garder les variables numerique et objet comme elles etaient, d'où le 'nr pour objet et 0 pour numerique (je n'ai pas trouvé par quoi d'autre que 0 totu en gardant la variable en numerique)

df["neighborhood_overview"]=df["neighborhood_overview"].fillna("nr")
df["transit"]=df["transit"].fillna("nr")
df["interaction"]=df["interaction"].fillna("nr")
df["house_rules"]=df["house_rules"].fillna("nr")
df["host_about"]=df["host_about"].fillna("nr")
df["amenities"]=df["amenities"].fillna("nr")
df["bathrooms"]=df["bathrooms"].fillna(0)
df["bedrooms"]=df["bedrooms"].fillna(0)
df["beds"]=df["beds"].fillna(0)
df["weekly_price"]=df["weekly_price"].fillna(0)
df["monthly_price"]=df["monthly_price"].fillna(0)
df["cleaning_fee"]=df["cleaning_fee"].fillna(0)

#Si on change d'avis concernant ces deux variables
# Et nous remplaçons les Nan de 'host_response_rate' par la moyenne, car 0 aurait un trop grand impact et 'nr' transformera la variable en 'objet'
df["host_response_rate"]=df["host_response_rate"].fillna(df["host_response_rate"].mean())
df["host_response_time"]=df["host_response_time"].fillna("nr")

#On vérifie si il reste des Nans.
print('\n Données manquantes par colonne :\n')
print(df.isna().sum())

#Non tout est bon, on peut passer à la suite

df.info()

#On fait une sauvegarde du dataset nettoyé
#df.to_pickle("airbnbcleaned.csv")

"""# **Exploration des variables**"""

#On stock les variables numérique dans num_df
num_df=df.select_dtypes(include="float64")

#On affiche la moyenne de chaque variable numerique dans un df appelé stats
stats = pd.DataFrame(num_df.mean(), columns = ['moyenne'])
stats.round(2)

#On remarque un soucis dans la variable 'maximum_nights'
#Et potentiellement des erreurs dans le 'minimum_nights' qui devrait etre proche de 1

"""

 Il semble y avoir des valeurs abérrantes dans maximum_nights pour avoir des valeurs si élevées

 Nous allons donc regarder cette variable ainsi que minimum_nights"""

#On recherche le bien avec la valeur la plus élevé pour maximum_nights
num_df['maximum_nights'].idxmax(axis = 0)

#On affiche le logement qui a la valeur la plus élevée pour maximum_nights
print(df.loc[642950])
#Nous constatons une valeur abérrante

#On affiche un autre bien qui contient des valeur abérrantes
print(df.loc[9112784])   #valeur aberrante minimum night 3888 et maximum night 9999

#print(df.loc[10192308 ])

#print(df.sort_values(by = 'maximum_nights').tail(40))

#print(df.sort_values(by = 'minimum_nights').tail(40))

"""A peu près 30 annonces ont une valeur abérrante pour la variable 'minimum_nights'

A peu près 80 annonces ont une valeur abérrante pour la variable 'maximum_nights'
"""

#On change les valeur aberrante (qui depasse 365jours) de maximum_nights en valeur de la variable 'availability_365', qui semble plus logique
num_df['maximum_nights'] = np.where(num_df['maximum_nights'] > 365, num_df['availability_365'], num_df['maximum_nights'])

#On verifie quel est le logement avec le plus grand nombre de maximum_nights
num_df['maximum_nights'].idxmax(axis = 0)

#On utilise cet index pour repérer le logement
print(df.loc[1372747])
#On constate qu'il a 365 donc plus de valeur aberrante dans cette variable

#On modifie les valeurs abérrantes de minimum_nights en remplaçant toutes les valeurs au dessus de 31 jours par 1
num_df.minimum_nights[num_df.minimum_nights > 31] = 1

#On verifie quel est le plus grand nombre de minimum_nights
num_df['maximum_nights'].idxmax(axis = 0)

#On affiche le logement avec l'index qui le plus grand nombre de minimum_night
print(df.loc[1372747])

#on ne remarque plus de valeur aberrante mais on remarque que ce logement a ausi le nombre de maximum_night maximum
#cet utilisateur loue donc pour des longues durées
# il a par ailleurs fait une erreur soit dans house_rules soit dans minimum_night car le nombre de nuitées minimum entrées ne correspond pas aux house_rules

#On refait la moyenne de chaque variable numerique dans un df appelé stats
stats2 = pd.DataFrame(num_df.mean(), columns = ['moyenne'])
stats2.round(2)

#On fait la mediane
stats2['median'] = num_df.median()
stats2

#On crée une variable mean_med_diff correspondant à la valeur absolue de la différence entre moyenne et médiane.
stats2['mean_med_diff'] = abs(stats2['moyenne'] - stats2['median'])
stats2.round(2)

#On ajoute les 3 colonnes qui correponde aux 3 quantiles de chaques variables numérique
stats2[['q1', 'q2', 'q3']] = num_df.quantile(q = [0.25,0.5,0.75]).transpose()

#On peut compléter notre analyse en s'intéressant au min, max de chaque variable. 
#La différence des deux nous donnera une idée de l'étendue sur laquelle se répartissent les valeurs.
stats2['min'] = num_df.min()
stats2['max'] = num_df.max()
stats2['min_max_diff'] = stats2['max'] - stats2['min']
stats2

"""On remarque que la médiane et la moyenne de la variable cible "review_scores_rating" sont très proche.

Il en est quasiment de même pour toutes les variables scores.

Ces variable sont donc hétérogènes.

Les variables les moins hétérogènes sont beds, host_total_listing_counts, host_response_rate, price, weekly_price, monthly_price et maximum_nights.

Pour la variable beds, on peut estimer que la moyenne se rapproche de 2 lits car il nous faut seulement des nombres entiers (un lit ne se divise pas).

Les résultats nous montrent que la moyenne est influencée par les valeurs extrêmes. Bien que la moyenne du nombre de lits par logement est de 2, il y a plus de logements avec un seul lit.

La différence entre la moyenne et la médiane de host_total_listing_count est très importante.

On en déduit que la moyenne est très impactée par les valeurs extrêmes et donc les propriétaires qui ont un usage commercial de Airbnb et mettent en lignes de nombreuses annnonces.

La valeur de la médiane, 1, nous informe que la grande majorité des propriétaires ne mettent qu'un logement en location.

La différence entre la moyenne et la médiane de la variable prix est assez importante. Le prix médian est bien inférieur au prix médian.

Les valeurs de weekly_price et monthly_price nous laisse supposer que peu de personne propose de prix à la semaine ou au mois.
"""

# tests de dépendance entre variables numériques:

#import pearsonr from scipy.stats
from scipy.stats.stats import pearsonr
#teste de correlation entre le nombre de review et la note globale d'une annonce
pd.DataFrame(pearsonr(df["number_of_reviews"],df["review_scores_rating"]),index=["pearson_coeff","p-value"],columns=["résultat_test"])
#la p-value > 5%, le coefficient proche de 0, il n'y a pas de corrélation entre les deux variables.

"""# **Analyse des variables catégorielles**"""

#On determine les variables catégorielles et on les stock dans cat_df
cat_df = df.select_dtypes(include=['O'])

#On affiche les variables categorielles
cat_df.head()

#host_since, first_review er last_review, n'ont rien a faire ici donc elles seront donc transformées en variables 'date time'

#On regarde la fréquence des modalités de la variable property_type
cat_df['property_type'].value_counts(normalize = True)

#Nous allons garder uniquement les types de propriété 'Apartment' et 'House' qui représente 99,6% du dataset
cat_df['property_type']=cat_df['property_type'].loc[cat_df['property_type'].isin(['Apartment','House'])]

#On supprime les ligne qui ne sont pas des appartement ou maison (devenu Nans avec le code précédent) 
cat_df=cat_df.dropna(axis=0,subset=['property_type'])

#On regarde la fréquence des modalités de la variable room_type
cat_df['room_type'].value_counts(normalize = True)

#Nous allons garder uniquement les types de chambre 'Entire home/apt' et 'Private room' qui représente 99,9% du dataset
cat_df['room_type']=cat_df['room_type'].loc[cat_df['room_type'].isin(['Entire home/apt','Private room'])]

#On supprime les ligne qui ne sont pas des 'Entire home/apt' ou 'Private room' (devenu Nans avec le code précédent) 
cat_df=cat_df.dropna(axis=0,subset=['room_type'])

#on regarde la fréquence des modalites de la variable city
cat_df['city'].value_counts(normalize = True)

"""beaucoup de variables catégorielles sont remplies librement par les propriétaires et donc quasiment toutes différentes.
Il conviendra donc de faire du text mining pour les analyser ultérieurement. 
"""

#On ré-assemble le dataframe variable numerique + variable catégorielles
df=pd.concat([num_df,cat_df],axis=1)

# Mettons maintenant ces variables au format datetime

df['host_since'] = pd.to_datetime(df['host_since'])
df['first_review'] = pd.to_datetime(df['first_review'])
df['last_review'] = pd.to_datetime(df['last_review'])

#Maintenant le que le df est reconstitué, on supprime les lignes qui contiennent des Nans du fait des changements dans cat_df
df=df.dropna(axis=0)

#On affiche un df.info
df.info()
#Tout semble en ordre pour passer à la visualisation

#on divise le df par villes puis on les regroupe dans df pour ecraser les données précédentes de df_london et df_paris
df_paris=df[df.city=='Paris']
df_london=df[df.city=='London']
#On supprimme la première ligne de df_london car elle pose problème pour la visualisation par la suite et elle contient beaucoup de valeurs "Non-renseignées"
df_london.drop( df_london.index[0], inplace=True)
df = df_paris.append(df_london)

#On fait une sauvegarde du dataset nettoyé
#df.to_pickle("airbnbcleaned.csv")

#Ouverture avec le code suivant
#df = pd.read_pickle('airbnbcleaned.csv')

"""# **Convertion variables catégorielles en numerique pour les inclurent dans notre modèle**"""

#On crée une colonne review_months qui contient le nombre de mois entre first_review et last_review
df['review_months'] = 12 * (df.last_review.dt.year - df.first_review.dt.year) + (df.last_review.dt.month - df.first_review.dt.month)

#On verifie quel est la date max
df['last_review'].idxmax(axis = 0)

#On affiche la date de la dernière review que l'on concidèrera comme date du jour 2017-04-06
print(df.loc['12582056'])

df['today'] = pd.Timestamp('20170406')

#On crée une colonne host_since_months qui contient le nombre de mois entre sur airbnb
df['host_since_months'] = 12 * (df.today.dt.year - df.host_since.dt.year) + (df.today.dt.month - df.host_since.dt.month)

#On supprimme la colonne today qui n'a plus d'importance
del df['today']

df.loc[df.review_scores_rating <96,'result']='<96'
df.loc[df.review_scores_rating >=96,'result']='>=96'

df.info()

#On supprime les colonnes qui n'ont pas d'importance pour le modele de prediction
df = df.drop(df.columns[[2,3,19,27,28,29,30,31,32,33,34,41,42,43]], axis=1)

df.info()

df.head()

X = df.drop(columns='result')
y = df['result']

X_encoded = pd.get_dummies(X, drop_first=True)
X_encoded.head()

"""# **Prédictions DecisionTree**"""

# Commented out IPython magic to ensure Python compatibility.
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score, roc_curve, auc
# %matplotlib inline
import warnings
warnings.filterwarnings('ignore')

X_train,X_test,y_train,y_test = train_test_split(X_encoded,y,test_size=0.2,stratify =y)

tree_clf = DecisionTreeClassifier(max_depth= 3)
tree_clf.fit(X_train,y_train)

print("Validation Mean F1 Score: ",cross_val_score(tree_clf,X_train,y_train,cv=5,scoring='f1_macro').mean())
print("Validation Mean Accuracy: ",cross_val_score(tree_clf,X_train,y_train,cv=5,scoring='accuracy').mean())

y_pred = tree_clf.predict(X_test)
print(classification_report(y_test, y_pred))

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

lr = LogisticRegression(solver='liblinear', max_iter=1000)

lr.fit(X_train_scaled, y_train)

print("Validation Mean F1 Score: ",cross_val_score(lr,X_train,y_train,cv=5,scoring='f1_macro').mean())
print("Validation Mean Accuracy: ",cross_val_score(lr,X_train,y_train,cv=5,scoring='accuracy').mean())

X_test_scaled = scaler.transform(X_test)
y_pred = lr.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

feat_importances = pd.Series(tree_clf.feature_importances_, index=X_train.columns)
feat_importances.nlargest(10).plot(kind = 'barh');

n = X_train.shape[1]

pca = PCA(n_components=2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

X_new = pca.fit_transform(X_train_scaled)

coeff = pca.components_.transpose()

xs = X_new[:,0]
ys = X_new[:,1]
scalex = 1.0/(xs.max() - xs.min())
scaley = 1.0/(ys.max() - ys.min())

principalDf = pd.DataFrame({'PC1': xs*scalex, 'PC2':ys * scaley})

y_train_pred = lr.predict(X_train_scaled)
finalDF = pd.concat([principalDf, pd.Series(y_train_pred, name='income')], axis = 1)

plt.figure(figsize=(30,25))

sns.scatterplot(x='PC1', y='PC2', hue= 'income', data = finalDF, alpha = 0.5);

for i in range(n):
    plt.arrow(0, 0, coeff[i,0]*1.5, coeff[i,1]*1.5,color = 'k',alpha = 0.5, head_width=0.01, )
    plt.text(coeff[i,0]*1.5, coeff[i,1] *1.5, X_train.columns[i], color = 'k')
    
plt.xlim(-0.6,0.8)
plt.ylim(-0.8,0.8);

from sklearn.tree import plot_tree

plt.figure(figsize=(15,12))
plot_tree(tree_clf, feature_names = X_train.columns.tolist(), filled=True);

#!pip install shap

import shap
import xgboost as xgb

# Encodage de la variable cible en 0/1
y_train = [0 if x == '<96' else 1 for x in y_train]
y_test = [0 if x == '<96' else 1 for x in y_test]

# Paramètres d'entraînement
params = {'objective': 'binary:logistic', 'max_depth': 100}

# Co,version des jeux de données en DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest =  xgb.DMatrix(data = X_test, label = y_test)

# Entraînement du modèle
bst = xgb.train(params, dtrain, 200)

probs = bst.predict(dtest)
preds = [0 if x<0.5 else 1 for x in probs]

print(classification_report(y_test, preds))

"""# **Predictions RandomForest**"""

#On lance un model de random forest pour essayer d'améliorer le model

from sklearn import model_selection
from sklearn import ensemble
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import VotingClassifier

#on ajuste X_train
scaler = preprocessing.StandardScaler().fit(X_train)

X_train_scaled = scaler.transform(X_train)
#on applique la meme transformation à X_test
X_test_scaled = scaler.transform(X_test)

# Création du classificateur et construction du modèle sur les données d'entraînement
clf_rf = ensemble.RandomForestClassifier() 

#On entraine l'algorithme sur l'ensemble d'entraînement (X_train_scaled et y_train).
clf_rf.fit(X_train_scaled, y_train)

#On crée un dictionnaire parametres contenant les valeurs possibles prises pour le paramètre
params_rf=[{'min_samples_split': [(i) for i in range (1,100,1)], 
                 'max_features': ['sqrt', 'log2']}]

#On applique la fonction model_selection.GridSearchCV() au modèle clf
grid_clf_rf = model_selection.GridSearchCV(estimator=clf_rf, param_grid=params_rf)

#On entraîne grid_clf sur l'ensemble d'entraînement, (X_train_scaled, y_train) 
#ET on sauvegarde les résultats dans l'objet grille.

grille_rf = grid_clf_rf.fit(X_train_scaled,y_train)

#on affiche toutes les combinaisons possibles d'hyperparamètres 
#et la performance moyenne du modèle associé par validation croisée.
print(pd.DataFrame.from_dict(grille_rf.cv_results_).loc[:,['params', 'mean_test_score']])

#on affiche la meilleur combinaison
print('\n''le meilleur paramètre est:',(grid_clf_rf.best_params_))

#Prédiction des features test et création de la matrice de confusion
y_pred = grid_clf_rf.predict(X_test_scaled)
pd.crosstab(y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite'])

#on affiche le score obtenu avec ce modèle
print("score:",grid_clf_rf.score(X_test_scaled,y_test))












































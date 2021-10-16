from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import altair as alt
import seaborn as sns
sns.set_style("whitegrid")
import base64
import datetime
from matplotlib import rcParams
from  matplotlib.ticker import PercentFormatter

header = st.beta_container()
dataset = st.beta_container()
features = st.beta_container()
model_training = st.beta_container()
st.set_option('deprecation.showPyplotGlobalUse', False)


@st.cache
def get_profile_pic():
	return Image.open('logo-airbnb.png')

def get_table_download_link(df):
		"""Generates a link allowing the data in a given panda dataframe to be downloaded
		in:  dataframe
		out: href string
		"""
		csv = df.to_csv(sep='\t', decimal=',', index=False, header=False)
		b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
		href = f'<a href="data:file/csv;base64,{b64}">Download</a>'
		return href


with dataset:
	df = pd.read_csv("data/airbnb.csv", engine="python", sep=';', quotechar='"', error_bad_lines=False)
	st.write(df.head())
	df_paris= pd.read_csv("data/airbnb_paris.csv", engine="python", sep=';', quotechar='"', error_bad_lines=False)
	df_london= pd.read_csv("data/airbnb_london.csv", engine="python", sep=';', quotechar='"', error_bad_lines=False)
	

	################################ SIDEBAR ################################

	st.sidebar.image(get_profile_pic(), use_column_width=False, width=250)
	st.sidebar.header("Bienvenue!")

	st.sidebar.markdown(" ")
	st.sidebar.markdown("*Nous sommes 4 étudiants en Datasciences chez Datascientest et nous travaillons sur ce projet Airbnb afin de valider notre diplome de Data Analyst.*")

	st.sidebar.markdown("**Author**: Valentin Goudey")


	st.sidebar.markdown("**Version:** 1.0.0")


	################################ SOMMAIRE ################################

	st.title("Airbnb listings Data Analyse")
	st.markdown('-----------------------------------------------------')

	st.markdown("**Dataset d’origine:** *Dataset airbnb rassemblant des données de logements dans de grandes villes tout autour du monde via le site opendatasoft*")
	st.markdown("*Dataset airbnb rassemblant des données de logements dans de grandes villes tout autour du monde via le site opendatasoft*")
	st.markdown("**Problématiques:** *Quels sont les critères impactant le plus la satisfaction d’un utilisateur Airbnb?*")
	st.markdown("*Quels sont les critères impactant le plus la satisfaction d’un utilisateur Airbnb?*")
	st.markdown("*Quels sont les critères pour obtenir le statut de 'SuperHost'?*")
	st.markdown("**Objectifs:**")
	st.markdown("*Identifier les caractéristiques propres à ce dataset et en déduire des informations pertinentes sur les relations entre les variables*")
	st.markdown("*Déterminer quelles sont les variables qui influencent le plus la variable Review Scores Rating*")
	st.markdown("*Pouvons prédire l'accès au statut de 'SuperHost'?*")
	st.markdown("**Variable cible:**")
	st.markdown("*Review score Rating (score global attribué à chaque logement par airbnb en fonction de 6 notes données par les utilisateurs)?*")
	st.markdown("*Creation d'une variable cible pour le statut de 'SuperHost'?*")

	st.header("Sommaire")

	st.markdown("Airbnb est une plateforme qui offre et guide la possibilité de relier deux groupes - les hôtes et les invités. N’importe qui avec une salle ouverte ou un espace libre peut fournir des services sur Airbnb à la communauté mondiale. C’est une bonne façon de fournir un revenu supplémentaire avec un minimum d’effort. C’est un moyen facile de promouvoir l’espace, parce que la plate-forme a du trafic et une base mondiale d’utilisateurs pour le soutenir. Airbnb offre aux hôtes un moyen facile de monétiser l’espace qui serait gaspillé.")

	st.markdown("D’autre part, nous avons des clients avec des besoins très spécifiques - certains peuvent être à la recherche d’un hébergement abordable à proximité des attractions de la ville, tandis que d’autres sont un appartement de luxe à la mer. Ils peuvent être des groupes, des familles ou des individus locaux et étrangers. Après chaque visite, les clients ont la possibilité de noter et de rester avec leurs commentaires. Nous allons essayer de savoir ce qui contribue à la popularité de la liste et de prédire si la liste a le potentiel de devenir l’un des rare 'SuperHost' en fonction de ses attributs.")

	st.markdown('-----------------------------------------------------')

	st.header("Airbnb  Listings: Data Analyse")
	st.markdown("Voici les 10 premiers enregistrements de données Airbnb. Ces dossiers sont regroupés sur 34 colonnes avec une variété d’informations comme la description du bien, son emplacement, le prix, le type de chambre, le minimum de nuits,le nombre de commentaires et ses commoditées.")
	st.markdown("Nous commencerons par nous familiariser avec les colonnes de l’ensemble de données, afin de comprendre ce que chaque caractéristique représente. Cela est important, car une mauvaise compréhension des caractéristiques pourrait nous amener à commettre des erreurs dans l’analyse des données et le processus de modélisation.")

	st.dataframe(df.head(10))

	st.markdown("Un autre point à propos de nos données est qu’il permet de trier la dataframe en cliquant sur n’importe quel en-tête de colonne, c’est une façon plus flexible de commander des données pour les visualiser.")

	#################### DISTRIBUTION GEOGRAPHIQUE ######################

	st.header("Liste des biens")
	st.markdown("Ci-dessous, nous soulignons la répartition géographique des inscriptions. Initialement, nous pouvons les filtrer par gamme de prix, le nombre minimum de nuits disponibles et le nombre d’avis, donc plus de flexibilité est ajoutée lorsque vous cherchez un endroit. ")
	st.markdown("Nous pourrions également filtrer par liste **prix**, **nuits minimales** sur une liste ou un minimum de **commentaires** reçus. ")

	#values = st.slider("Tranche de prix (€)", float(df.price.min()), float(df.price.clip(upper=100000.).max()), (500., 1500.))
	#min_nights_values = st.slider('Minimum de nuits', 0, 30, (1))
	#reviews = st.slider('Minimum de commentaires', 0, 700, (0))
	#st.map(df.query(f"price.between{values} and minimum_nights<={min_nights_values} and number_of_reviews>={reviews}")[["latitude", "longitude"]].dropna(how="any"), zoom=10)

	st.markdown("D’une manière générale, la carte montre que les emplacements dans le centre-ville sont plus chers, tandis que les périphéries sont moins chères. En outre, le centre-ville semble avoir son propre modèle.")
	

	#################### centre d'interet ######################

	st.header("Qu’est-ce que tu cherches ?")
	st.write(f"Dans les colonnes {df.shape[1]}, vous pourriez vouloir afficher uniquement un sous-ensemble.")
	st.markdown("_**Note:** Il est possible de filtrer nos données de manière plus conventionnelle en utilisant les caractéristiques suivantes : **Prix**, **Type de chambre**, **Minimum de nuits**, **Quartier***, **Description de l'hote**, **Commentaires**")
	defaultcols = ["price", "minimum_nights", "room_type", "neighbourhood_cleansed", "description", "number_of_reviews"]
	cols = st.multiselect('', df.columns.tolist(), default=defaultcols)
	st.dataframe(df[cols].head(10))


	################################## Quartier ###############################
	

	st.header("Quartiers")
	st.markdown("La ville de Paris comprend 20 divisions de quartiers.")


	fig = sns.scatterplot(df_paris.longitude,df_paris.latitude,hue=df_paris.neighbourhood_cleansed).set_title('Paris')
	st.pyplot()

	st.markdown("La ville de  Londres en comprends 33.")

	fig = sns.scatterplot(df_london.longitude,df_london.latitude,hue=df_london.neighbourhood_cleansed).set_title('Londres')
	st.pyplot()


	################### Pourcentage de distribution par quartiers #####################

	st.header("Disponibilitée et distribution par quartier de Paris")
	st.markdown("La variable **availability_365** indique le nombre de jours dans l'année (365) où le bien est disponible.")

	neighborhood_paris = st.radio("Quartiers", df_paris.neighbourhood_cleansed.unique())
	is_expensive_paris = st.checkbox("Bien les plus chères Paris")
	is_expensive_paris = " and price<100" if not is_expensive_paris else ""
	
	@st.cache
	def get_availability_paris(show_exp, neighborhood):
		return df_paris.query(f"""neighbourhood_cleansed==@neighborhood{is_expensive_paris}\
			and availability_365>0""").availability_365.describe(\
				percentiles=[.1, .25, .5, .75, .9, .99]).to_frame().T

	st.table(get_availability_paris(is_expensive_paris, neighborhood_paris))

	st.header("Disponibilitée et distribution par quartier de Londres")
	st.markdown("La variable **availability_365** indique le nombre de jours dans l'année (365) où le bien est disponible.")

	neighborhood_london = st.radio("Quartiers", df_london.neighbourhood_cleansed.unique())
	is_expensive_london = st.checkbox("Bien les plus chères Londres")
	is_expensive_london = " and price<100" if not is_expensive_london else ""
	
	@st.cache
	def get_availability_london(show_exp, neighborhood):
		return df_london.query(f"""neighbourhood_cleansed==@neighborhood{is_expensive_london}\
			and availability_365>0""").availability_365.describe(\
				percentiles=[.1, .25, .5, .75, .9, .99]).to_frame().T

	st.table(get_availability_london(is_expensive_london, neighborhood_london))

if __name__ == '__main__':
	main()
	
	###################### Nombre de Type de chambre par Qaurtiers #######################

	st.markdown("Following, let's check the relationship between property type and neighbourhood. The primary question we aim to answer is whether different boroughs constitute of different rental types. Though in the expanded dataset there are more than 20 types, we will be focussing on the top 4 by their total count in the city and understanding their distribution in each borough.")

	room_types_df = df_paris.groupby(['neighbourhood_cleansed', 'room_type']).size().reset_index(name='Quantity')
	room_types_df = room_types_df.rename(columns={'neighbourhood_cleansed': 'District', 'room_type':'Room Type'})
	room_types_df['Percentage'] = room_types_df.groupby(['District'])['Quantity'].apply(lambda x:100 * x / float(x.sum()))

	sns.set_style("whitegrid")
	sns.set(rc={'figure.figsize':(11.7,8.27)})
	fig = sns.catplot(y='Percentage', x='District', hue="Room Type", data=room_types_df, height=6, kind="bar", palette="muted", ci=95);
	fig.set(ylim=(0, 100))


	for ax in fig.axes.flat:
			ax.yaxis.set_major_formatter(PercentFormatter(100))
	plt.show()

	st.pyplot()

	st.markdown("The plot shows the ratio of property type and the total number of properties in the borough.")

	st.subheader("Some key observations from the graph are:")

	st.markdown(" - We can see that **Private Room** listings are highest in number in all tree borough except Manhattan and Staten Island. Staten Island has more ‘House’ style property than ‘Apartments’ thus, probably the only possible listings are apartments. This analysis seems intuitive, as we know that Staten Island is not that densely populated and has a lot of space.")

	st.markdown(" - The maximum **Entire home/apt** listings are located in Manhattan, constituting 60.55% of all properties in that neighborhood. Next is Staten Island with 49.86% **Entire home/apt**.")

	st.markdown(" - Queens, Brooklyn and the Bronx also have many listings for **Private room**. In Queens, 59.25% of the apartments are of the **Private room** type, which is larger than in the Bronx.")

	st.markdown(" - **Shared Room** listings types are also common in New York. Bronx constitutes of 5.59% of **Shared Room** listings type followed by Queens with 3.58% **Shared Room** listings type.")

	st.markdown(" - Manhattan has nearly 1.55% of **Hotel Room** listings. Next is Queens with 6.83% **Hotel Room** listings followed by Brooklyn with 3.32%. The other tree borough does not present any **Hotel Room** listings.")


	###################### PRICE AVERAGE BY ACOMMODATION #########################

	st.header("Average price by room type")

	st.markdown("To listings based on room type, we can show price average grouped by borough.")

	avg_price_room = df.groupby("room_type").price.mean().reset_index()\
			.round(2).sort_values("price", ascending=False)\
			.assign(avg_price=lambda x: x.pop("price").apply(lambda y: "%.2f" % y))

	avg_price_room = avg_price_room.rename(columns={'room_type':'Room Type', 'avg_price': 'Average Price ($)', })

	st.table(avg_price_room)

	st.markdown("Despite together **Hotel Room** listings represent just over 10%, they are responsible for the highest price average, followed by **Entire home/apt**. Thus there are a small number of **Hotel Room** listings due its expensive prices.")


	############################ MOST RATED HOSTS #############################

	st.header("Most rated hosts")

	rcParams['figure.figsize'] = 15,7
	ranked = df.groupby(['host_name'])['number_of_reviews'].count().sort_values(ascending=False).reset_index()
	ranked = ranked.head(5)
	sns.set_style("whitegrid")
	fig = sns.barplot(y='host_name', x='number_of_reviews', data=ranked,palette="Blues_d",)
	fig.set_xlabel("Nº de Reviews",fontsize=10)
	fig.set_ylabel("Host",fontsize=10)

	st.pyplot()

	st.write(f"""The host **{ranked.iloc[0].host_name}** is at the top with {ranked.iloc[0].number_of_reviews} reviews.
	**{ranked.iloc[1].host_name}** is second with {ranked.iloc[1].number_of_reviews} reviews. It should also be noted that reviews are not positive or negative reviews, but a count of feedbacks provided for the accommodation.""")


	#################### DEMAND AND PRICE ANALYIS ######################

	st.header("Demand and Price Analysis")

	st.markdown("In this section, we will analyse the demand for Airbnb listings in New York City. We will look at demand over the years since the inception of Airbnb in 2010 and across months of the year to understand seasonlity. We also wish to establish a relation between price and demand. The question we aspire to answer is whether prices of listings fluctuate with demand. We will also conduct a more granular analysis to understand how prices vary by days of the week.")
	st.markdown("To study the demand, since we did not have data on the bookings made over the past year, we will use **number of reviews** variable as the indicator for demand. As per Airbnb, about 50% of guests review the hosts/listings, hence studying the number of review will give us a good estimation of the demand.")

	accommodation = st.radio("Room Type", df.room_type.unique())

	all_accommodation = st.checkbox('All Accommodations')

	demand_df = df[df.last_review.notnull()]
	demand_df.loc[:,'last_review'] = pd.to_datetime(demand_df.loc[:,'last_review'])
	price_corr_df = demand_df

	if all_accommodation:
		demand_df = df[df.last_review.notnull()]
		demand_df.loc[:,'last_review'] = pd.to_datetime(demand_df.loc[:,'last_review'])
	else:
		demand_df = demand_df.query(f"""room_type==@accommodation""")

	fig = px.scatter(demand_df, x="last_review", y="number_of_reviews", color="room_type")
	fig.update_yaxes(title="Nª Reviews")
	fig.update_xaxes(title="Last Review Dates")
	st.plotly_chart(fig)

	st.markdown("The number of unique listings receiving reviews has increased over the years. Highly rated locations also tend to be the most expensive ones. We can see an almost exponential increase in the number of reviews, which as discussed earlier, indicates an exponential increase in the demand.")

	st.markdown("But about the price ? We also can show the same plot, but this time we take into account the **price** feature along years. Again we use **last review dates** to modeling time series in order to achieve a proportion between price over the years. Let's check it out.")

	fig = px.scatter(price_corr_df, x="last_review", y="price", color="neighbourhood_group")
	fig.update_yaxes(title="Price ($)")
	fig.update_xaxes(title="Last Review Dates")
	st.plotly_chart(fig)

	st.markdown("The price smoothly increases along the years if we take into account the number of reviews according the borough. Sightly Manhattan it's most expensive borough followed by Brooklyn, some listings apear also as outliers in past 2 years. Let's take a look again in the number of reviews, but this time we group by boroughs.")

	fig = px.scatter(price_corr_df, x="last_review", y="number_of_reviews", color="neighbourhood_group")
	fig.update_yaxes(title="Nª Reviews")
	fig.update_xaxes(title="Last Review Dates")
	st.plotly_chart(fig)

	st.markdown("The number of reviews for Queens appears more often. We get some insights here. 1)  the room type most sought in Queens is the **private room** (as seen in the previous plot). 2)  the price range in Queens is below Manhattan, so perhaps the Queens contemplate the _\"best of both worlds\"_ being the most cost-effective district.")

	st.markdown("But there is some correlation between reviews increase and prices? Let's check it out.")

	fig = px.scatter(price_corr_df, y="price", x="number_of_reviews", color="neighbourhood_group")
	fig.update_xaxes(title="Nª Reviews")
	fig.update_yaxes(title="Price ($)")
	st.plotly_chart(fig)

	st.markdown("Well, actually does not happens a correlation between reviews and price apparently, the cheaper the more opinions he has. Another point is,  Queens has more reviews than others, which reinforces our theory about being the most cost-effective district.")


	st.header("Most Rated Listings")
	st.markdown("We can slide to filter a range of numbers in the sidebar to view properties whose review count falls in that range.")

	reviews = st.slider('', 0, 12000, (100))

	df.query(f"number_of_reviews<={reviews}").sort_values("number_of_reviews", ascending=False)\
	.head(50)[["number_of_reviews", "price", "neighbourhood", "room_type", "host_name"]]

	st.write("654 is the highest number of reviews and only a single property has it. In general, listings with more than 400 reviews are priced below $ 100,00. Some are between $100,00 and $200,00, and only one is priced above $200,00.")


	############################# PRICE DISTRIBUTION ###########################

	st.header("Price Distribution")

	st.markdown("Bellow we can select a custom price range from the side bar to update the histogram below and check the distribution skewness.")
	st.write("""Select a custom price range from the side bar to update the histogram below.""")
	values = st.slider("Faixa de Preço", float(df.price.min()), float(df.price.clip(upper=1000.).max()), (50., 300.))
	f = px.histogram(df.query(f"price.between{values}"), x="price", nbins=100, title="Price distribution")
	f.update_xaxes(title="Price")
	f.update_yaxes(title="No. of listings")
	st.plotly_chart(f, color='lifeExp')

	@st.cache
	def get_availability(show_exp, neighborhood):
		return df.query(f"""neighbourhood_group==@neighborhood{show_exp}\
		and availability_365>0""").availability_365.describe(\
		percentiles=[.1, .25, .5, .75, .9, .99]).to_frame().T


	############################# CONCLUSIONS ###########################

	st.header("Conclusions")

	st.markdown("Through this exploratory data analysis and visualization project, we gained several interesting insights into the Airbnb rental market. Below we will summarise the answers to the questions that we wished to answer at the beginning of the project:")

	st.markdown("**How do prices of listings vary by location? What localities in NYC are rated highly by guests?** Manhattan has the most expensive rentals compared to the other boroughs. Prices are higher for rentals closer to city hotspots. Rentals that are rated highly on the location by the host also have higher prices")

	st.markdown("**How does the demand for Airbnb rentals fluctuate across the year and over years?** In general, the demand (assuming that it can be inferred from the number of reviews) for Airbnb listings has been steadily increasing over the years.")

	st.markdown("**Are the demand and prices of the rentals correlated?** Average prices of the rentals increase across the year, which correlates with demand.")

	st.header("Limitations")

	st.markdown(" - We did not have data for past years and hence could not compare current rental trends with past trends. Hence, there was an assumption made, particularly in the demand and supply section of the report to understand the booking trends.")

	st.markdown(" Below the data used in this research is available to reproducible research.")

	st.markdown(get_table_download_link(df), unsafe_allow_html=True)

	################################## FOOTER ##################################

	st.markdown('-----------------------------------------------------')
	st.text('Developed by Valentin Goudey - 2020')
	

if __name__ == '__main__':
	main()


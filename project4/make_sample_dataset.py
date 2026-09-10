"""
Builds a small stand-in dataset with the SAME COLUMN SCHEMA as the
IMDB 5000 Movie Dataset (Kaggle). This lets us build/test the full
pipeline (feature extraction -> preference model -> study interface)
without the original file. Swapping in the real movie_metadata.csv
later requires no code changes elsewhere, only re-running this
extraction step on the real file.

Columns follow the well-known IMDB5000 schema (subset of the most
relevant fields for this project).
"""
import pandas as pd

movies = [
    # title, genres, director, actor1, actor2, actor3, duration, year, score, lang, country, rating, budget, gross, votes
    ("The Shawshank Redemption","Drama","Frank Darabont","Tim Robbins","Morgan Freeman","Bob Gunton",142,1994,9.3,"English","USA","R",25000000,28341469,2500000),
    ("The Godfather","Crime|Drama","Francis Ford Coppola","Marlon Brando","Al Pacino","James Caan",175,1972,9.2,"English","USA","R",6000000,246120974,1800000),
    ("The Dark Knight","Action|Crime|Drama","Christopher Nolan","Christian Bale","Heath Ledger","Aaron Eckhart",152,2008,9.0,"English","USA","PG-13",185000000,1004558444,2700000),
    ("Pulp Fiction","Crime|Drama","Quentin Tarantino","John Travolta","Uma Thurman","Samuel L. Jackson",154,1994,8.9,"English","USA","R",8000000,213928762,2000000),
    ("Forrest Gump","Drama|Romance","Robert Zemeckis","Tom Hanks","Robin Wright","Gary Sinise",142,1994,8.8,"English","USA","PG-13",55000000,678226465,2000000),
    ("Inception","Action|Adventure|Sci-Fi","Christopher Nolan","Leonardo DiCaprio","Joseph Gordon-Levitt","Elliot Page",148,2010,8.8,"English","USA","PG-13",160000000,836848102,2200000),
    ("Fight Club","Drama","David Fincher","Brad Pitt","Edward Norton","Helena Bonham Carter",139,1999,8.8,"English","USA","R",63000000,101209702,2000000),
    ("The Matrix","Action|Sci-Fi","Lana Wachowski","Keanu Reeves","Laurence Fishburne","Carrie-Anne Moss",136,1999,8.7,"English","USA","R",63000000,463517383,1800000),
    ("Goodfellas","Biography|Crime|Drama","Martin Scorsese","Robert De Niro","Ray Liotta","Joe Pesci",146,1990,8.7,"English","USA","R",25000000,46836394,1100000),
    ("The Lord of the Rings: The Fellowship of the Ring","Adventure|Drama|Fantasy","Peter Jackson","Elijah Wood","Ian McKellen","Orlando Bloom",178,2001,8.8,"English","New Zealand","PG-13",93000000,871530324,1800000),
    ("Se7en","Crime|Drama|Mystery","David Fincher","Morgan Freeman","Brad Pitt","Kevin Spacey",127,1995,8.6,"English","USA","R",33000000,327311859,1600000),
    ("The Silence of the Lambs","Crime|Drama|Thriller","Jonathan Demme","Jodie Foster","Anthony Hopkins","Lawrence A. Bonney",118,1991,8.6,"English","USA","R",19000000,272742922,1400000),
    ("Saving Private Ryan","Drama|War","Steven Spielberg","Tom Hanks","Matt Damon","Tom Sizemore",169,1998,8.6,"English","USA","R",70000000,481840909,1300000),
    ("Interstellar","Adventure|Drama|Sci-Fi","Christopher Nolan","Matthew McConaughey","Anne Hathaway","Jessica Chastain",169,2014,8.6,"English","USA","PG-13",165000000,701729206,1700000),
    ("The Green Mile","Crime|Drama|Fantasy","Frank Darabont","Tom Hanks","Michael Clarke Duncan","David Morse",189,1999,8.6,"English","USA","R",60000000,286801374,1300000),
    ("Gladiator","Action|Adventure|Drama","Ridley Scott","Russell Crowe","Joaquin Phoenix","Connie Nielsen",155,2000,8.5,"English","USA","R",103000000,460583960,1400000),
    ("The Departed","Crime|Drama|Thriller","Martin Scorsese","Leonardo DiCaprio","Matt Damon","Jack Nicholson",151,2006,8.5,"English","USA","R",90000000,291465034,1200000),
    ("Whiplash","Drama|Music","Damien Chazelle","Miles Teller","J.K. Simmons","Melissa Benoist",106,2014,8.5,"English","USA","R",3300000,13092000,900000),
    ("The Prestige","Drama|Mystery|Sci-Fi","Christopher Nolan","Christian Bale","Hugh Jackman","Scarlett Johansson",130,2006,8.5,"English","USA","PG-13",40000000,109676311,1300000),
    ("The Lion King","Animation|Adventure|Drama","Roger Allers","Matthew Broderick","Jeremy Irons","James Earl Jones",88,1994,8.5,"English","USA","G",45000000,422783777,900000),
    ("Titanic","Drama|Romance","James Cameron","Leonardo DiCaprio","Kate Winslet","Billy Zane",194,1997,7.9,"English","USA","PG-13",200000000,2187463944,1100000),
    ("Avatar","Action|Adventure|Fantasy|Sci-Fi","James Cameron","Sam Worthington","Zoe Saldana","Sigourney Weaver",162,2009,7.8,"English","USA","PG-13",237000000,2787965087,1300000),
    ("Frozen","Animation|Adventure|Comedy|Family|Fantasy|Musical","Chris Buck","Kristen Bell","Idina Menzel","Jonathan Groff",102,2013,7.4,"English","USA","PG",150000000,1274219009,600000),
    ("Toy Story","Animation|Adventure|Comedy|Family|Fantasy","John Lasseter","Tom Hanks","Tim Allen","Don Rickles",81,1995,8.3,"English","USA","G",30000000,373554033,900000),
    ("Finding Nemo","Animation|Adventure|Comedy|Family","Andrew Stanton","Albert Brooks","Ellen DeGeneres","Alexander Gould",100,2003,8.2,"English","USA","G",94000000,940335536,900000),
    ("Up","Animation|Adventure|Comedy|Drama|Family","Pete Docter","Edward Asner","Jordan Nagai","John Ratzenberger",96,2009,8.3,"English","USA","PG",175000000,735099082,900000),
    ("La La Land","Comedy|Drama|Music|Musical|Romance","Damien Chazelle","Ryan Gosling","Emma Stone","John Legend",128,2016,8.0,"English","USA","PG-13",30000000,446000000,500000),
    ("The Notebook","Drama|Romance","Nick Cassavetes","Ryan Gosling","Rachel McAdams","James Garner",123,2004,7.8,"English","USA","PG-13",29000000,115603229,500000),
    ("Pretty Woman","Comedy|Romance","Garry Marshall","Richard Gere","Julia Roberts","Jason Alexander",119,1990,7.0,"English","USA","R",14000000,463406268,300000),
    ("Bridesmaids","Comedy|Romance","Paul Feig","Kristen Wiig","Maya Rudolph","Rose Byrne",125,2011,6.8,"English","USA","R",32500000,288383261,300000),
    ("Superbad","Comedy","Greg Mottola","Jonah Hill","Michael Cera","Christopher Mintz-Plasse",113,2007,7.6,"English","USA","R",20000000,169958411,500000),
    ("The Hangover","Comedy","Todd Phillips","Bradley Cooper","Ed Helms","Zach Galifianakis",100,2009,7.7,"English","USA","R",35000000,467483912,700000),
    ("Get Out","Horror|Mystery|Thriller","Jordan Peele","Daniel Kaluuya","Allison Williams","Bradley Whitford",104,2017,7.7,"English","USA","R",4500000,255407225,500000),
    ("A Quiet Place","Drama|Horror|Sci-Fi","John Krasinski","Emily Blunt","John Krasinski","Millicent Simmonds",90,2018,7.5,"English","USA","PG-13",17000000,340900469,300000),
    ("Hereditary","Drama|Horror|Mystery","Ari Aster","Toni Collette","Alex Wolff","Milly Shapiro",127,2018,7.3,"English","USA","R",10000000,79323375,300000),
    ("It","Drama|Horror","Andy Muschietti","Jaeden Martell","Bill Skarsgard","Finn Wolfhard",135,2017,7.3,"English","USA","R",35000000,701842551,400000),
    ("The Conjuring","Horror|Mystery|Thriller","James Wan","Vera Farmiga","Patrick Wilson","Ron Livingston",112,2013,7.5,"English","USA","R",20000000,319494638,400000),
    ("John Wick","Action|Crime|Thriller","Chad Stahelski","Keanu Reeves","Michael Nyqvist","Alfie Allen",101,2014,7.4,"English","USA","R",20000000,88761661,600000),
    ("Mad Max: Fury Road","Action|Adventure|Sci-Fi","George Miller","Tom Hardy","Charlize Theron","Nicholas Hoult",120,2015,8.1,"English","Australia","R",150000000,375775505,900000),
    ("Die Hard","Action|Thriller","John McTiernan","Bruce Willis","Alan Rickman","Bonnie Bedelia",132,1988,8.2,"English","USA","R",28000000,140767956,800000),
    ("Skyfall","Action|Adventure|Thriller","Sam Mendes","Daniel Craig","Judi Dench","Javier Bardem",143,2012,7.8,"English","UK","PG-13",200000000,1108561013,700000),
    ("Black Panther","Action|Adventure|Sci-Fi","Ryan Coogler","Chadwick Boseman","Michael B. Jordan","Lupita Nyong'o",134,2018,7.3,"English","USA","PG-13",200000000,1346913161,600000),
    ("The Avengers","Action|Adventure|Sci-Fi","Joss Whedon","Robert Downey Jr.","Chris Evans","Scarlett Johansson",143,2012,8.0,"English","USA","PG-13",220000000,1518815515,1200000),
    ("Guardians of the Galaxy","Action|Adventure|Comedy|Sci-Fi","James Gunn","Chris Pratt","Zoe Saldana","Dave Bautista",121,2014,8.0,"English","USA","PG-13",170000000,772776600,900000),
    ("Spirited Away","Animation|Adventure|Family|Fantasy","Hayao Miyazaki","Daveigh Chase","Suzanne Pleshette","Miyu Irino",125,2001,8.6,"Japanese","Japan","PG",19000000,395580000,700000),
    ("Parasite","Comedy|Drama|Thriller","Bong Joon Ho","Song Kang-ho","Lee Sun-kyun","Cho Yeo-jeong",132,2019,8.6,"Korean","South Korea","R",11400000,258773645,700000),
    ("Amelie","Comedy|Romance","Jean-Pierre Jeunet","Audrey Tautou","Mathieu Kassovitz","Rufus",122,2001,8.3,"French","France","R",10000000,174200000,700000),
    ("City of God","Crime|Drama","Fernando Meirelles","Alexandre Rodrigues","Leandro Firmino","Matheus Nachtergaele",130,2002,8.6,"Portuguese","Brazil","R",3300000,7563415,700000),
    ("Oldboy","Action|Drama|Mystery","Park Chan-wook","Choi Min-sik","Yoo Ji-tae","Kang Hye-jeong",120,2003,8.4,"Korean","South Korea","R",3000000,15000000,500000),
    ("The Grand Budapest Hotel","Adventure|Comedy|Crime|Drama","Wes Anderson","Ralph Fiennes","F. Murray Abraham","Mathieu Amalric",99,2014,8.1,"English","Germany","R",25000000,174800000,700000),
    ("Eternal Sunshine of the Spotless Mind","Drama|Romance|Sci-Fi","Michel Gondry","Jim Carrey","Kate Winslet","Tom Wilkinson",108,2004,8.3,"English","USA","R",20000000,74036040,900000),
    ("No Country for Old Men","Crime|Drama|Thriller","Ethan Coen","Tommy Lee Jones","Javier Bardem","Josh Brolin",122,2007,8.1,"English","USA","R",25000000,171627166,900000),
    ("There Will Be Blood","Drama","Paul Thomas Anderson","Daniel Day-Lewis","Paul Dano","Ciaran Hinds",158,2007,8.2,"English","USA","R",25000000,76166675,600000),
    ("Her","Drama|Romance|Sci-Fi","Spike Jonze","Joaquin Phoenix","Amy Adams","Scarlett Johansson",126,2013,8.0,"English","USA","R",23000000,48267157,600000),
    ("Blade Runner 2049","Drama|Mystery|Sci-Fi","Denis Villeneuve","Ryan Gosling","Harrison Ford","Ana de Armas",164,2017,8.0,"English","USA","R",150000000,259263944,500000),
    ("Dunkirk","Action|Drama|History|Thriller|War","Christopher Nolan","Fionn Whitehead","Tom Glynn-Carney","Jack Lowden",106,2017,7.8,"English","UK","PG-13",100000000,526940665,600000),
    ("1917","Drama|War","Sam Mendes","Dean-Charles Chapman","George MacKay","Daniel Mays",119,2019,8.2,"English","UK","R",95000000,384574433,500000),
    ("Joker","Crime|Drama|Thriller","Todd Phillips","Joaquin Phoenix","Robert De Niro","Zazie Beetz",122,2019,8.4,"English","USA","R",55000000,1074251311,1200000),
    ("Knives Out","Comedy|Crime|Drama|Mystery","Rian Johnson","Daniel Craig","Chris Evans","Ana de Armas",130,2019,7.9,"English","USA","PG-13",40000000,311922683,500000),
]

cols = ["movie_title","genres","director_name","actor_1_name","actor_2_name","actor_3_name",
        "duration","title_year","imdb_score","language","country","content_rating",
        "budget","gross","num_voted_users"]

df = pd.DataFrame(movies, columns=cols)
df.to_csv("movie_metadata_sample.csv", index=False)
print(f"Wrote {len(df)} movies, columns: {list(df.columns)}")
print(df.head(3).to_string())

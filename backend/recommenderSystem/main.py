import pandas as pd

class Helpers:
    @staticmethod
    def extractGenres(animeDataframe):
        genres = animeDataframe['Genres'].str.split(', ').tolist()
        genresList = [genre for sublist in genres for genre in sublist]
        uniqueGenres = set(genresList)
        return uniqueGenres
    
    @staticmethod
    def extractTypes(animeDataframe):
        typesList = animeDataframe['Type']
        uniqueTypes = set(typesList)
        return uniqueTypes
    
    @staticmethod
    def extractRatingCategories(animeDataframe):
        ratingsList = animeDataframe["Rating"]
        uniqueRatings = set(ratingsList)
        return uniqueRatings
    
    @staticmethod
    def extractProducers(animeDataframe):
        producers = animeDataframe["Producers"].str.split(', ').tolist()
        producersList = [producer for sublist in producers for producer in sublist]
        uniqueProducers = set(producersList)
        return uniqueProducers
    
    @staticmethod
    def extractLicensors(animeDataframe):
        licensors = animeDataframe["Licensors"].str.split(", ").tolist()
        licensorsList = [licensor for sublist in licensors for licensor in sublist]
        uniqueLicensors = set(licensorsList)
        return uniqueLicensors

    @staticmethod
    def extractStudios(animeDataframe):
        studios = animeDataframe["Studios"].str.split(", ").tolist()
        studiosList = [studio for sublist in studios for studio in sublist]
        uniqueStudios = set(studiosList)
        return uniqueStudios

    @staticmethod
    def extractSources(animeDataframe):
        sourcesList = animeDataframe["Source"]
        uniqueSources = set(sourcesList)
        return uniqueSources
    
    @staticmethod
    def extractNames(animeDataframe):
        namesList = animeDataframe["Name"].tolist()
        return namesList
    
    @staticmethod
    def extractEnglishNames(animeDataframe):
        englishNamesList = animeDataframe["English name"].tolist()
        uniqueEnglishNames = set(englishNamesList)
        return uniqueEnglishNames
    
    @staticmethod
    def extractJapaneseNames(animeDataframe):
        japaneseNamesList = animeDataframe["Japanese name"].tolist()
        uniqueJapeneseNames = set(japaneseNamesList)
        return uniqueJapeneseNames
    
class AnimeRecommenderSystem:
    def __init__(self):
        self.animeDataset = pd.read_csv("./Dataset/anime.csv")
        self.synopsisDataset = pd.read_csv("./Dataset/anime_with_synopsis.csv")
        self.animeGenres = Helpers.extractGenres(self.animeDataset)
        self.dataMap = {
            "animeTypes": Helpers.extractTypes(self.animeDataset),
            "ratingCategories": Helpers.extractRatingCategories(self.animeDataset),
            "producers": Helpers.extractProducers(self.animeDataset),
            "licensors": Helpers.extractLicensors(self.animeDataset),
            "studios": Helpers.extractStudios(self.animeDataset),
            "sources": Helpers.extractSources(self.animeDataset)  
        }
        self.animeTypes = None
        self.ratingCategories = None
        self.producers = None
        self.licensors = None
        self.studios = None
        self.sources = None
        self.names = None

    def activateFilters(self, filters):
        for filter in filters:
            setattr(self, filter, self.dataMap[filter])

    def deactivateFilters(self, filters):
        for filter in filters:
            setattr(self, filter, None)

    def synopsisRecommender(self):
        pass

if __name__ == "__main__":
    print("lol")
    ARS = AnimeRecommenderSystem()
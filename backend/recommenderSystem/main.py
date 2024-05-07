import pandas as pd
from itertools import combinations

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
    
    @staticmethod
    def extractEpisodeIntervals(animeDataframe):
        intervalCount = 5
        animeDataframe['Episodes'] = pd.to_numeric(animeDataframe['Episodes'], errors='coerce')
        minimumEpisodes = animeDataframe['Episodes'].min()
        maximumEpisodes = animeDataframe['Episodes'].max()
        intervalWidth = (maximumEpisodes - minimumEpisodes) / intervalCount
        intervals = []
        for i in range(intervalCount):
            lowerBound = minimumEpisodes + i * intervalWidth
            upperBound = lowerBound + intervalWidth
            intervals.append(f"{upperBound:.0f}")
        return intervals
        
    @staticmethod
    def extractPremieredTime(animeDataframe):
        premiereTimes = animeDataframe["Premiered"].tolist()
        uniquePremiereTimes = set(premiereTimes)
        return uniquePremiereTimes
    
    @staticmethod
    def createUserAnimeMatrix(userPreferenceInformation):
        userAnimeMatrix = {'animeID': userPreferenceInformation["userAnimeIDs"], 'rating': userPreferenceInformation["userRatings"]}
        userAnimeMatrix = pd.DataFrame(userAnimeMatrix)
        userAnimeMatrix.set_index('animeID', inplace=True)
        return userAnimeMatrix

    @staticmethod
    def createUserProfileMatrix(userPreferenceInformation, animeDataframe, contentAttributes):
        userProfileMatrix = pd.DataFrame(0, index=userPreferenceInformation["userAnimeIDs"], columns=list(contentAttributes[0]))
        for animeID in userPreferenceInformation["userAnimeIDs"]:
            animeGenres = animeDataframe[animeDataframe['MAL_ID'] == animeID]['Genres'].str.split(', ').tolist()[0]
            animeProducers = animeDataframe[animeDataframe['MAL_ID'] == animeID]['Producers'].str.split(', ').tolist()[0]
            animeLicensors = animeDataframe[animeDataframe['MAL_ID'] == animeID]['Licensors'].str.split(', ').tolist()[0]
            animeStudios = animeDataframe[animeDataframe['MAL_ID'] == animeID]['Studios'].str.split(', ').tolist()[0]
            userProfileMatrix.loc[animeID, animeGenres] = 1
            userProfileMatrix.loc[animeID, animeProducers] = 1
            userProfileMatrix.loc[animeID, animeLicensors] = 1
            userProfileMatrix.loc[animeID, animeStudios] = 1
        userProfileMatrix = userProfileMatrix.fillna(0)
        return userProfileMatrix

    @staticmethod
    def scaleUserProfileMatrix(userAnimeMatrix, userProfileMatrix):
        scaledUserProfileMatrix = userAnimeMatrix.values * userProfileMatrix.values
        scaledUserProfileMatrix = pd.DataFrame(scaledUserProfileMatrix, columns=userProfileMatrix.columns, index=userAnimeMatrix.index)
        return scaledUserProfileMatrix

    @staticmethod
    def normalizeUserProfileMatrix(scaledUserProfileMatrix, userProfileMatrix):
        columnSums = scaledUserProfileMatrix.sum(axis=0)
        normalizedUserProfileMatrix = columnSums / columnSums.sum()
        normalizedUserProfileMatrix.index = userProfileMatrix.columns
        return normalizedUserProfileMatrix

    @staticmethod
    def createUnratedAnimeMatrix(animeDataframe, contentAttributes):
        columns = [element for set_item in contentAttributes for element in set_item]
        unratedAnimeMatrix = pd.DataFrame(0, index=animeDataframe['MAL_ID'], columns=columns)
        return unratedAnimeMatrix

    #unratedAnimeMatrix.drop(userPreferenceInformation['userAnimeIDs'], inplace=True)
    
    @staticmethod
    def createUnratedOneHotEncodingMatrix(unratedAnimeMatrix, animeDataframe, filters):
        for animeID in unratedAnimeMatrix.index:
            if "animeGenres" in filters:
                animeGenres = animeDataframe[animeDataframe['MAL_ID'] == animeID]['Genres'].str.split(', ').tolist()[0]
                unratedAnimeMatrix.loc[animeID, animeGenres] = 1
            if "producers" in filters:
                animeProducers = animeDataframe[animeDataframe['MAL_ID'] == animeID]['Producers'].str.split(', ').tolist()[0]
                unratedAnimeMatrix.loc[animeID, animeProducers] = 1
            if "licensors" in filters:
                animeLicensors = animeDataframe[animeDataframe['MAL_ID'] == animeID]['Licensors'].str.split(', ').tolist()[0]
                unratedAnimeMatrix.loc[animeID, animeLicensors] = 1
            if "studios" in filters:
                animeStudios = animeDataframe[animeDataframe['MAL_ID'] == animeID]['Studios'].str.split(', ').tolist()[0]
                unratedAnimeMatrix.loc[animeID, animeStudios] = 1
        return unratedAnimeMatrix


class AnimeRecommenderSystem:
    def __init__(self):
        self.animeDataset = pd.read_csv("./Dataset/anime.csv")
        self.synopsisDataset = pd.read_csv("./Dataset/anime_with_synopsis.csv")
        self.animeGenres = Helpers.extractGenres(self.animeDataset)
        self.producers = Helpers.extractProducers(self.animeDataset)
        self.licensors = Helpers.extractLicensors(self.animeDataset)
        self.studios = Helpers.extractStudios(self.animeDataset)
        self.dataMap = [
            "animeGenres",
            "producers",
            "licensors",
            "studios"
        ]

    def permuteAndCreateUnratedAnimeMatrices(self):
        categories = self.dataMap
        powerSet = []
        for r in range(1, len(categories) + 1):
            combinationsList = combinations(categories, r)
            for combination in combinationsList:
                powerSet.append("-".join(combination))
        for classifiers in powerSet:
            classifier = classifiers.split("-")
            filteredList = [getattr(self, filter) for filter in classifier]
            unratedAnimeMatrix = Helpers.createUnratedAnimeMatrix(self.animeDataset, filteredList)
            oneHotEncodedUnratedMatrix = Helpers.createUnratedOneHotEncodingMatrix(unratedAnimeMatrix, self.animeDataset, classifier)
            oneHotEncodedUnratedMatrix.to_csv("./UnratedEncodings/"+str(classifiers)+".csv")
    
    def contentRecommender(self, userPreferenceInformation, filteredAnimeIDs, filters, relevantResultsCount=10):
        userAnimeMatrix = Helpers.createUserAnimeMatrix(userPreferenceInformation)
        self.activateFilters(filters)
        filteredList = [getattr(self, filter) for filter in filters]
        userProfileMatrix = Helpers.createUserProfileMatrix(userPreferenceInformation, self.animeDataset, filteredList)
        scaledUserProfileMatrix = Helpers.scaleUserProfileMatrix(userAnimeMatrix, userProfileMatrix)
        normalizedUserProfileMatrix = Helpers.normalizeUserProfileMatrix(scaledUserProfileMatrix, userProfileMatrix)
        # unratedAnimeMatrix = Helpers.createUnratedAnimeMatrix(self.animeDataset,userPreferenceInformation, filteredList)
        # Helpers.createUnratedOneHotEncodingMatrix(unratedAnimeMatrix, self.animeDataset)
        self.deactivateFilters(filters)

    def synopsisRecommender(self):
        pass

if __name__ == "__main__":
    ARS = AnimeRecommenderSystem()
    # ARS.contentRecommender({"userAnimeIDs":[1,5,6], "userRatings":[1,2,3]},"",["studios","animeGenres"])
    ARS.permuteAndCreateUnratedAnimeMatrices()
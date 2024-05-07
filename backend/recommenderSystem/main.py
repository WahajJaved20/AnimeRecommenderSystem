import pandas as pd
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import PorterStemmer

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
    def createUserProfileMatrix(userPreferenceInformation, animeDataframe, contentAttributes, filters):
        columns = [element for set_item in contentAttributes for element in set_item]
        userProfileMatrix = pd.DataFrame(0, index=userPreferenceInformation["userAnimeIDs"], columns=columns)
        
        for animeID in userPreferenceInformation["userAnimeIDs"]:
            if "animeGenres" in filters:
                animeGenres = animeDataframe[animeDataframe['MAL_ID'] == animeID]['Genres'].str.split(', ').tolist()[0]
                userProfileMatrix.loc[animeID, animeGenres] = 1
            if "producers" in filters:
                animeProducers = animeDataframe[animeDataframe['MAL_ID'] == animeID]['Producers'].str.split(', ').tolist()[0]
                userProfileMatrix.loc[animeID, animeProducers] = 1
            if "licensors" in filters:
                animeLicensors = animeDataframe[animeDataframe['MAL_ID'] == animeID]['Licensors'].str.split(', ').tolist()[0]
                userProfileMatrix.loc[animeID, animeLicensors] = 1
            if "studios" in filters:
                animeStudios = animeDataframe[animeDataframe['MAL_ID'] == animeID]['Studios'].str.split(', ').tolist()[0]
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

    @staticmethod
    def createUnratedUserAnimeMatrix(filters, userPreferenceInformation):
        fileName = "-".join(filters)
        fileName += ".csv"
        unratedAnimeMatrix = pd.read_csv("./UnratedEncodings/"+fileName)
        unratedAnimeMatrix.drop(userPreferenceInformation['userAnimeIDs'], inplace=True)
        unratedAnimeMatrix = unratedAnimeMatrix.drop("MAL_ID", axis = 1)
        return unratedAnimeMatrix

            
class VectorSpaceModel:
    def __init__(self):
        self.sypnopsisDataset = pd.read_csv("./Dataset/anime_with_synopsis.csv")
        self.stopwordsList = []
        self.postingList = {}
        self.tokens = []
        self.documentFrequency = {}

    def readStopwordsFile(self, filepath):
        with open(filepath, "r") as file:
            lines = file.readlines()
        self.stopwordsList = [line.rstrip() for line in lines]
        return lines

    def isStopword(self, token):
        if token in self.stopwordsList:
            return True
        else:
            return False
    
    def normalizeToken(self, token):
        normalizedToken = ""
        for char in token:
            if char.isalnum():
                normalizedToken += char
        return normalizedToken
    
    def casefoldToken(self, token):
        return token.casefold()
    
    def hasNumber(self, token):
        for char in token:
            if char.isdigit():
                return True
        return False
    
    def processSypnopsis(self):
        stemmer = PorterStemmer()
        sypnopsis = self.sypnopsisDataset["sypnopsis"]
        for storyNumber in range(len(sypnopsis)):
            story = sypnopsis[storyNumber]
            markedTokens = []
            for token in story:
                if not self.isStopword(token) and not self.hasNumber(token):
                    token = self.normalizeToken(token)
                    if token == "":
                        continue
                    token = self.casefoldToken(token)
                    token = stemmer.stem(token)
                    if token not in self.postingList.keys:
                        self.tokens.append(token)
                    if not token in markedTokens:
                        if not token in self.documentFrequency.keys:
                            self.documentFrequency[token] = 1
                        else:
                            self.documentFrequency[token] += 1
                        markedTokens.append(token)
                    found = False
                    for i in range(len(self.postingList[token])):
                        if self.postingList[token][i] == storyNumber:
                            found = True
                            self.postingList[token][i][1] += 1
                    if not found:
                        if len(self.postingList[token] == 0):
                            self.postingList[token] = [[storyNumber, 1]]
                        else:
                            self.postingList[token].append([storyNumber, 1])
            self.tokens.sort()

class AnimeRecommenderSystem:
    def __init__(self):
        self.animeDataset = pd.read_csv("./Dataset/anime.csv")
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
    
    def contentRecommender(self, userPreferenceInformation, filters, relevantResultsCount=10):
        userAnimeMatrix = Helpers.createUserAnimeMatrix(userPreferenceInformation)
        filteredList = [getattr(self, filter) for filter in filters]
        userProfileMatrix = Helpers.createUserProfileMatrix(userPreferenceInformation, self.animeDataset, filteredList, filters)
        scaledUserProfileMatrix = Helpers.scaleUserProfileMatrix(userAnimeMatrix, userProfileMatrix)
        normalizedUserProfileMatrix = Helpers.normalizeUserProfileMatrix(scaledUserProfileMatrix, userProfileMatrix)
        unratedUserAnimeMatrix = Helpers.createUnratedUserAnimeMatrix(filters, userPreferenceInformation)
        cosineSimilarities = cosine_similarity([normalizedUserProfileMatrix], unratedUserAnimeMatrix)
        animeCosineSimilarities = pd.Series(cosineSimilarities[0], index=unratedUserAnimeMatrix.index)
        topCosineSimilarities = animeCosineSimilarities.nlargest(relevantResultsCount)
        return topCosineSimilarities

    def sypnopsisRecommender(self):
        VSM = VectorSpaceModel()
        VSM.readStopwordsFile("./VSMUtils/Stopword-List.txt")
        VSM.processSypnopsis()

if __name__ == "__main__":
    nltk.download('punkt')
    ARS = AnimeRecommenderSystem()
    # ARS.contentRecommender({"userAnimeIDs":[1,5,6], "userRatings":[1,2,3]},["animeGenres","studios"])
    # ARS.permuteAndCreateUnratedAnimeMatrices()
    ARS.sypnopsisRecommender()
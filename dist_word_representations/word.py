import numpy as np
from scipy.stats import spearmanr

def calculateC(window_size, v, v_c):
    upperC = [[0 for k in range(len(v_c))] for l in range(len(v))] # C matrix
    with open("wiki-1percent.txt", "r") as file3: #reading wikipedia corpus Note: change wiki file to 1 percent for analysis 
        sentences = file3.readlines() 
        for sentence in sentences:
            sentence = "<s> " + sentence + " </s>"
            sentence = sentence.split()
            for word_index, word in enumerate(sentence):
                if word in v:
                    i_index = v[word]
                    for index in range(max(word_index - window_size, 0), min(word_index + window_size + 1, len(sentence))):
                        if index == word_index:
                            continue
                        if sentence[index] in v_c:
                            context_word_index = v_c[sentence[index]]
                            upperC[i_index][context_word_index] += 1 
    return upperC              

def evaluate(C):
    with open("cs402_assn1/men.txt", "r") as file4: # reading men file
        lines = file4.readlines()[1:]
        eval_score = []
        human_score = []
        for line in lines:
            line = line.split()
            w1 = line[0]
            w2 = line[1]
            score = float(line[2])
            if w1 in v and w2 in v:
                i = v[w1]
                j = v[w2]
                x = np.linalg.norm(C[i])
                y = np.linalg.norm(C[j])
                if x == 0 or y == 0:
                    continue
                else:
                    human_score.append(score)
                    eval_score.append(np.dot(C[i], C[j]) / (x * y))
        print("comparison  based on of MEN file:")
        print(spearmanr(human_score, eval_score))

    with open("cs402_assn1/simlex-999.txt", "r") as file5: #reading simlex
        lines = file5.readlines()[1:]
        eval_score = []
        human_score = []
        for line in lines:
            line = line.split()
            w1 = line[0]
            w2 = line[1]
            score = float(line[2])
            if w1 in v and w2 in v:
                i = v[w1]
                j = v[w2]
                x = np.linalg.norm(C[i])
                y = np.linalg.norm(C[j])
                if x == 0 or y == 0:
                    continue
                else:
                    human_score.append(score)
                    eval_score.append(np.dot(C[i], C[j]) / (x * y)) #cosine similarity
        print("comparison based on SIMLEX:")
        print(spearmanr(human_score, eval_score))

def calculate_PMI(M):
    total_sum = np.sum(M)
    row_sums = np.sum(M, axis=1)
    col_sums = np.sum(M, axis=0)
    num_rows, num_cols = M.shape

    C = np.zeros((num_rows, num_cols))

    # Calculate PMI for each word pair
    for i in range(num_rows):
        for j in range(num_cols):
            if M[i, j] == 0:
            
                C[i, j] = 0
            else:
                joint_prob = M[i, j] / total_sum
                word1_prob = row_sums[i] / total_sum
                word2_prob = col_sums[j] / total_sum
                PMI = np.log2(joint_prob / (word1_prob * word2_prob))
                C[i, j] = PMI
    return C

def k_nearest_neighbor(query_index, C, range):
    # Two arrays
    array1 = C[query_index]
    similarity_arr = []

    for row in C:
        # Calculate dot product
        dot_product = np.dot(array1, row)

        # Calculate magnitudes
        magnitude_array1 = np.linalg.norm(row)
        magnitude_array2 = np.linalg.norm(array1)
        if magnitude_array1 == 0 or magnitude_array2 == 0:
            continue
        # Calculate cosine similarity
        similarity = dot_product / (magnitude_array1 * magnitude_array2)
        similarity_arr.append(similarity)
    
    sorted_indices = np.argsort(similarity_arr)
    return sorted_indices[:(range+1)]
    

if __name__ == "__main__":
    
    with open('cs402_assn1/vocab-wordsim.txt', 'r') as file1: #reading vocab file
        lines = file1.readlines()
    v = {}
    i = 0
    for line in lines:
        line_clean = line.strip() 
        v[line_clean] = i
        i += 1
    # print(v)

    with open('cs402_assn1/vocab-25k.txt', 'r') as file2: #reading context vocab file
        lines = file2.readlines()
    v_c = {}
    j = 0
    for line in lines:
        line_clean = line.strip()
        v_c[line_clean] = j
        j += 1
    # print(v_c)
    
    window_size = [1, 3, 6]
    for window in window_size:    
        print(f"calculating for window size of {window}:")
        upperC = calculateC(window, v, v_c)
        
         #Distributional Counting
        print("correlation for C:")
        evaluate(upperC)

         #Computing PMIs
        print("correlation for C_pmi:")
        upperC_np = np.array(upperC)
        pmi = calculate_PMI(upperC_np)
        evaluate(pmi)

    window_size = [1, 6]
    n = 10
    for size in window_size:
        print(f"calculating for context window {size}:")
        v_c_copy = v_c
        upperC = calculateC(size, v_c, v_c_copy)
        
        upperC_np = np.array(upperC)
        pmi = calculate_PMI(upperC_np)

        word_array = ["monster", "bank", "cell", "apple", "apples", "axes", "frame", "light", "well"]
        for elem in word_array:    
            print(f"calculating {n} nearest neighbors for {elem}:")
            query_index = v_c[elem]
            arr = k_nearest_neighbor(query_index, pmi, n)
            word_arr = []
            for word in arr:
                words = [key for key, v in v_c.items() if v == word]
                word_arr.append(words)
            print(word_arr[1:])
import math
import csv
import random
import pandas as pd
import matplotlib.pyplot as plt
orb_list = []
ssim_list = []
vgg16_list = []
grade_list = []

with open('train_data.csv', 'r', encoding='utf-8-sig') as file:
    reader = csv.DictReader(file, delimiter=';')
    
    # Print the headers to debug
    print("CSV Headers:", reader.fieldnames)
    
    for row in reader:
        # Print the row to debug
        ##print("Row:", row)
        
        orb_list.append(float(row['ORB'].strip()))
        ssim_list.append(float(row['SSIM'].strip()))
        vgg16_list.append(float(row['VGG16'].strip()))
        grade_list.append(int(row['GRADED'].strip()))

"""ORB_input = 0.813725490196078  ##bunları da input olarak al
SSIM_input = 0.732770371531803
VGG16_input = 0.593429148197174"""
"""ORB_input = float(input("ORB değerini giriniz "))
SSIM_input = float(input("SSIM değerini giriniz "))
VGG16_input = float(input("VGG16 değerini giriniz "))"""


ORB_input = []
SSIM_input = []
VGG16_input = []
GRADED_input = []
with open('test_data.csv', 'r', encoding='utf-8-sig') as fileT:
    readerT = csv.DictReader(fileT, delimiter=';')
    
    # Print the headers to debug
    print("CSV Headers:", readerT.fieldnames)
    
    for row in readerT:
        # Print the row to debug
        ##print("Row:", row)
        
        ORB_input.append(float(row['ORB'].strip()))
        SSIM_input.append(float(row['SSIM'].strip()))
        VGG16_input.append(float(row['VGG16'].strip()))
        GRADED_input.append(float(row['GRADED'].strip()))

def  euclidean_distance(ORB_train, SSIM_train, VGG16_train, orb_test, ssim_test, vgg16_test,  index):
    x_axis = (orb_test[index] - ORB_train)**2
    y_axis = (ssim_test[index] - SSIM_train)**2
    z_axis = (vgg16_test[index] - VGG16_train)**2
    
    distance = math.sqrt(x_axis + y_axis + z_axis)
    ##print(distance)
    return distance
##euclidean_distance(2,-8,3)

##distance_list = []

"""orb_list = [0.762376237623762,
0.757894736842105,
0.710526315789474,
0.642857142857143,
0.91025641025641,
0.953703703703704,
0.780952380952381,
0.724137931034483,
0.819277108433735,
0.435897435897436,
0.841584158415842,
0.804878048780488,
0.780487804878049,
0.684782608695652,
0.863636363636364,
0.46875,
0.463414634146341,
0.728070175438597,
0.0,
0.813084112149533,
0.740740740740741,
0.785123966942149,
0.839285714285714,
0.885057471264368,
0.91566265060241,
0.0,
0.697247706422018,
0.752941176470588,
0.807692307692308,
0.844155844155844,
0.878787878787879,
0.790909090909091,
0.698630136986301,
0.808988764044944,
0.640776699029126,
0.897959183673469,
0.834782608695652,
0.912280701754386,
0.627659574468085,
0.752688172043011,
0.747252747252747,
0.731707317073171,
0.868421052631579,
0.663461538461538,
0.732558139534884,
0.615384615384615,
0.473684210526316,
0.762886597938144,
0.89247311827957,
0.621621621621622,
0.770642201834862,
0.862068965517241,
0.649484536082474,
0.559322033898305,
0.80952380952381,
0.88034188034188,
0.772277227722772,
0.70873786407767,
0.862385321100917,
0.834862385321101,
0.647058823529412,
0.590909090909091,
0.814159292035398, 
0.731958762886598,
0.722689075630252,
0.824175824175824,
0.896907216494845,
0.716666666666667,
0.847619047619048,
0.707547169811321,
0.877551020408163,
0.693069306930693,
0.806451612903226,
0.89010989010989,
0.489583333333333,
0.901960784313726,
0.764705882352941,
0.646464646464647,
0.733333333333333,
0.656565656565657,
0.79047619047619,
0.635294117647059,
0.517647058823529,
0.838383838383838,
0.924731182795699,
0.868686868686869,
0.703703703703704,
0.950980392156863,
0.806451612903226,
0.793103448275862,
0.708333333333333,
0.450450450450451,
0.701030927835051,
0.93,
0.538461538461538,
0.780952380952381,
0.77027027027027,
0.7,
0.846846846846847,
0.620689655172414]

ssim_list = [0.739084445713922,
0.652588571810367,
0.709609858512839,
0.381175731204338,
0.821069311476337,
0.779727822129295,
0.762232793382061,
0.746845851929277,
0.809799920609481,
0.31148039362798,
0.728119864993552,
0.775994749139086,
0.754409762827398,
0.761364884157976,
0.730249258576838,
0.512557193065459,
0.375508486318931,
0.745396060508597,
0.878692267186314,
0.790863688536436,
0.599189218455362,
0.553507031567571,
0.734948451069797,
0.795214976563022,
0.756686547289532,
0.878576820331611,
0.36132527112331,
0.699197172989844,
0.749706417875358,
0.781083288832703,
0.736507670145113,
0.706153995758547,
0.7597580864104,
0.785448177208746,
0.651382678979988,
0.727188253494929,
0.70235139874222,
0.731365614257777,
0.738169972311061,
0.623783607237217,
0.756023241671703,
0.749863748897858,
0.783525016403939,
0.706127013943642,
0.765226829513727,
0.655175355804033,
0.723392017421739,
0.802105516064773,
0.700225166238836,
0.63814877315681,
0.760494152078705,
0.752606467303671,
0.758078821011202,
0.753529657048319,
0.712204022269671,
0.79352944194918,
0.783659539106624,
0.683408856363437,
0.797922935396984,
0.713633027342452,
0.70386854892536,
0.681110227768504,
0.746230323459381,
0.729834650098322,
0.71714082591313,
0.716893446043029,
0.714636303979344,
0.697013459499928,
0.712213627953889,
0.699696019707907,
0.661926595697042,
0.748139705882396,
0.744872017435968,
0.735688943567205,
0.746019883392861,
0.725997111176814,
0.723928213547805,
0.748258785887374,
0.729116802772548,
0.717133737620484,
0.693854104648428,
0.699603340791223,
0.735347500834492,
0.706386052550601,
0.7010716878775,
0.694945272452226,
0.761896871785281,
0.768972513598631,
0.665145236281191,
0.663699862635558,
0.745389479600292,
0.447472483766467,
0.714291464032125,
0.754109547653755,
0.562906472300554,
0.78097979799452,
0.675637507872522,
0.730858489443089,
0.727905003618631,
0.594824598006256]

vgg16_list = [0.601345300674438,
0.491634756326675,
0.577256739139557,
0.392715394496918,
0.594584286212921,
0.50813627243042,
0.658591866493225,
0.592464029788971,
0.541598260402679,
0.308961778879166,
0.570509493350983,
0.764566361904144,
0.566234350204468,
0.622986972332001,
0.650597035884857,
0.426337748765945,
0.431807070970535,
0.612075626850128,
0.463902354240417,
0.66511458158493,
0.505883514881134,
0.501420140266418,
0.58922153711319,
0.613329231739044,
0.525141298770905,
0.468186885118484,
0.355899065732956,
0.552354454994202,
0.681487202644348,
0.738212943077087,
0.589066743850708,
0.541246771812439,
0.674122273921967,
0.698370575904846,
0.541670858860016,
0.573942899703979,
0.58086234331131,
0.553816318511963,
0.654765129089355,
0.55930769443512,
0.544821739196777,
0.701724827289581,
0.599142789840698,
0.604598999023437,
0.673708915710449,
0.431824177503586,
0.596578598022461,
0.642499625682831,
0.589226126670837,
0.431698679924011,
0.665289580821991,
0.653864204883575,
0.666087210178375,
0.585199177265167,
0.59836757183075,
0.628220796585083,
0.517388463020325,
0.584725499153137,
0.635416865348816,
0.663031399250031,
0.600022614002228,
0.604308545589447,
0.638999402523041,
0.545651137828827,
0.493273109197617,
0.597066640853882,
0.635399460792542,
0.564575731754303,
0.575690388679504,
0.475161492824554,
0.541834115982056,
0.608423411846161,
0.584493398666382,
0.62964802980423,
0.402823328971863,
0.681383430957794,
0.538224995136261,
0.539815723896027,
0.61900532245636,
0.613691389560699,
0.650687873363495,
0.601425528526306,
0.568265795707703,
0.608230948448181,
0.649954497814178,
0.529156982898712,
0.648641347885132,
0.6043701171875,
0.621158719062805,
0.525466620922089,
0.601824700832367,
0.391401141881943,
0.584635019302368,
0.538543343544006,
0.465607434511185,
0.65899795293808,
0.684263288974762,
0.486749112606049,
0.656925618648529,
0.650681853294373]

grade_list = [72,
92,
92,
92,
40,
46,
92,
55,
10,
72,
74,
92,
82,
10,
82,
92,
23,
92,
5,
66,
56,
73,
62,
80,
46,
10,
73,
92,
92,
81,
92,
56,
82,
92,
90,
46,
92,
62,
44,
92,
92,
54,
92,
74,
90,
91,
92,
63,
82,
92,
82,
80,
80,
92,
92,
59,
73,
92,
92,
64,
92,
82,
70,
91,
91,
100,
82,
36,
92,
82,
77,
82,
92,
92,
50,
82,
80,
30,
92,
80,
92,
100,
92,
92,
82,
92,
82,
52,
92,
71,
92,
77,
92,
42,
92,
77,
92,
92,
92,
92]"""
"""for i in range(len(orb_list)):
    distance_list.append(euclidean_distance(orb_list[i], ssim_list[i], vgg16_list[i]))"""

k = 3 ##input olarak al

##sorted_distance_list = sorted(distance_list)

"""for i in range(len(sorted_distance_list)):
    print(sorted_distance_list[i])"""


def rearrangedGrade(sorted_list, distance_list, grade_list):
    reArranged = []

    for i in range(len(sorted_list)):
        first_index = 0
        for j in range(len(sorted_list)):
            if sorted_list[i] == distance_list[j]: 
                 first_index = j
                 break
             
        reArranged.append(grade_list[first_index])

    return reArranged

##grades_sorted_list = rearrangedGrade(sorted_distance_list, distance_list, grade_list)

"""for i in range(len(sorted_distance_list)):
    print(grades_sorted_list[i])"""

def weighted_grade(grades, distance, real_grades, index):
    around_grade = 0
    total_diverse_distance = 0
    for i in range(k):
        around_grade += grades[i] * (1/distance[i])
    for i in range(k):
        total_diverse_distance += (1/distance[i])
    around_grade = around_grade / total_diverse_distance
    print(around_grade, " ", real_grades[index])
    return around_grade
##weighted_grade(grades_sorted_list, sorted_distance_list)

def mean_absolute_deviation(weighted_grades, graded_test):
    total_difference = 0
    total_real_grades = 0
    for i in range(len(weighted_grades)):
        total_difference += abs(weighted_grades[i] - graded_test[i])
        total_real_grades += graded_test[i]

    result = total_difference / total_real_grades
    return result   

def accuracy(predicted_grades, real_grades):
    true_list = []
    false_list = []
    accuracy_value = 0
    for i in range(len(predicted_grades)):
        if(((predicted_grades[i] + predicted_grades[i]*0.22) >= real_grades[i]) and ((predicted_grades[i] - predicted_grades[i]*0.22) <= real_grades[i])):
            true_list.append(1)
            """print((predicted_grades[i] + predicted_grades[i]*0.16), " >= ", real_grades[i])
            print(((predicted_grades[i] - predicted_grades[i]*0.16), " <= ", real_grades[i]))"""

        else:
            false_list.append(0)
    print("***********************")
    print(true_list, " true list ", false_list, " false list ")
    accuracy_value = len(true_list) / len(predicted_grades)
    return accuracy_value

def cross_validation(orb_train, ssim_train, vgg16_train, graded_train):
    accuracy_list = []
    test_index = []
    weighted_grades = []

    orb_train_fold = []
    ssim_train_fold = []
    vgg16_train_fold = []
    graded_train_fold = []

    orb_test_list = []
    ssim_test_list = []
    vgg16_test_list = []
    graded_test_list = [] 
    distance_list = []

    
    for i in range(5):
        test_index = random.sample(range(0,len(orb_train) - 1),20)
        ##print(f"Random numbers for iteration {i+1}: {test_index}")
        for j in range(len(test_index)):
            orb_test_list.append(orb_train[test_index[j]])
            ssim_test_list.append(ssim_train[test_index[j]])
            vgg16_test_list.append(vgg16_train[test_index[j]])
            graded_test_list.append(graded_train[test_index[j]])
            ##print(orb_test_list[j], " ssim ", ssim_test_list[j], " vgg16 ", vgg16_test_list[j], " graded ", graded_test_list[j])
        for j in range(len(orb_train)):
            if j in test_index:
                continue
            else:
                orb_train_fold.append(orb_train[j])
                ssim_train_fold.append(ssim_train[j])
                vgg16_train_fold.append(vgg16_train[j])
                graded_train_fold.append(graded_train[j])
        for j in range(len(test_index)): 
            distance_list = []       
            for t in range(len(orb_train_fold)):
                 distance_list.append(euclidean_distance(orb_train_fold[t], ssim_train_fold[t], vgg16_train_fold[t], orb_test_list, ssim_test_list, vgg16_test_list, j))
           
            sorted_distance_list = sorted(distance_list)
            grades_sorted_list = rearrangedGrade(sorted_distance_list, distance_list, graded_train_fold)
            weighted_grades.append(weighted_grade(grades_sorted_list, sorted_distance_list, graded_test_list, j))
        accuracy_list.append(accuracy(weighted_grades, grades_sorted_list))
        print("mean absolute deviation : ", mean_absolute_deviation(weighted_grades, graded_test_list)) 
        print("accuracy ", accuracy(weighted_grades, grades_sorted_list))
        print("rmse ", root_mean_score_error(grades_sorted_list, weighted_grades))
        ##print("ara**************")    
        ##random_numbers = []
        orb_test_list = []
        ssim_test_list = []
        vgg16_test_list = []
        graded_test_list = []
        distance_list = []
        weighted_grades = []

        orb_train_fold = []
        ssim_train_fold = []
        vgg16_train_fold = []
        graded_train_fold = []        
    plot_accuracy_cross_validation(accuracy_list)

def plot_accuracy_cross_validation(accuracy_list):
    accuracy_values = accuracy_list
    cv_indices = range(1, len(accuracy_values) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(cv_indices, accuracy_values, marker='o', linestyle='-')
    plt.title('Cross-Validation Accuracy Values')
    plt.xlabel('Cross-Validation Fold')
    plt.ylabel('Accuracy')
    plt.xticks(cv_indices)  # x ekseni etiketleri her bir cross-validation katmanının indeksiyle eşleşecek şekilde ayarlanır
    plt.ylim(0, 1)  # Y ekseni aralığını (ylim) 0 ile 1 arasında belirleyin
    plt.grid(True)
    plt.show()


def root_mean_score_error(real_grade, predicted_grade):
    total = 0
    result = 0
    for i in range(len(predicted_grade)):
        total += abs(real_grade[i] - predicted_grade[i])**2
    print("total ", total)
    result = total / len(predicted_grade)
    print("result ", result)

    result = math.sqrt(result)
    print("result ", result)

    return result

def main():
    weighted_grades = []
    distance_list = [] 
    for i in range(len(ORB_input)): 
        distance_list = []       
        for j in range(len(orb_list)):
            distance_list.append(euclidean_distance(orb_list[j], ssim_list[j], vgg16_list[j], ORB_input, SSIM_input, VGG16_input, i))
           
        sorted_distance_list = sorted(distance_list)
        grades_sorted_list = rearrangedGrade(sorted_distance_list, distance_list, grade_list)
        weighted_grades.append(weighted_grade(grades_sorted_list, sorted_distance_list, GRADED_input, i))
        
    
    print("mean absolute deviation : ", mean_absolute_deviation(weighted_grades, GRADED_input))    
    cross_validation(orb_list,ssim_list,vgg16_list,grade_list)

main()            
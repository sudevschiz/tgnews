import sys
import task1_languages as t1
import task2_news as t2
import task3_categories as t3
import task4_threads as t4
import task5_top as t5
import json



if __name__ == "__main__":
    if sys.argv[1] == "languages":
        j_out = t1.compute(sys.argv[2])
#         print(j_out)
    elif sys.argv[1] == "news":
        print("Finding news articles in folder: "+sys.argv[2])
        j_out = t1.compute(sys.argv[2])
#         print(j_out)
    elif sys.argv[1] == "categories":
        print("Tagging categories in folder: "+sys.argv[2])
        j_out = t1.compute(sys.argv[2])
#         print(j_out)
    elif sys.argv[1] == "threads":
        print("Finding threads in folder: "+sys.argv[2])
        j_out = t1.compute(sys.argv[2])
#         print(j_out)
    elif sys.argv[1] == "top":
        print("Threads sorted by relevance in folder: "+sys.argv[2])
        j_out = t1.compute(sys.argv[2])
#         print(j_out)
    else:
        print("Unsupported arguments")
    
    print(json.dumps(j_out, indent=4, sort_keys=True))
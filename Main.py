import os

while True:
    choice = input("What would you like to do?\n1. Ask questions of the chatbot?\n2. Add/ask logic information of the chatbot?\n3. Analyze pictures with the chatbot?\n4. Exit the program.\n")
##############################
#Part1
############################## 
    if choice == "1":
        # Run program A if the user chooses option 1
        import xml.etree.ElementTree as ET
        import pandas as pd
        import nltk
        import re
        import sys
        import wikipedia
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import speech_recognition as sr
        import pyttsx3

        # load AIML XML file
        tree = ET.parse('mybot-basic.xml')
        aiml_root = tree.getroot()

        # define lemmatizer
        lemmatizer = nltk.stem.WordNetLemmatizer()

        # define tf-idf vectorizer
        vectorizer = TfidfVectorizer()

        # create a recognizer instance
        r = sr.Recognizer()

        # create a text-to-speech engine instance
        engine = pyttsx3.init()

        # start chatting
        while True:
            # ask the user to choose the input mode
            print("Please select the input mode: ")
            print("1. Voice")
            print("2. Text")
            choice = input()
            
            if choice == '1':
                # use microphone as source
                with sr.Microphone() as source:
                    # adjust for ambient noise levels
                    r.adjust_for_ambient_noise(source, duration=1)
                    # listen for the user's input
                    print("Please say something...")
                    audio = r.listen(source)

                try:
                    # use speech recognition to convert audio to text
                    user_input = r.recognize_google(audio)
                    print("You said:", user_input)

                # if speech recognition fails
                except sr.UnknownValueError:
                    print("I'm sorry, I didn't catch that. Please try again.")
                    engine.say("I'm sorry, I didn't catch that. Please try again.")
                    engine.runAndWait()
                    continue

            elif choice == '2':
                user_input = input("You: ")
            else:
                print("Invalid input. Please select either '1' or '2'.")
                continue
            
            if user_input.lower() == 'exit':
                print("Goodbye!")
                engine.say("Goodbye!")
                engine.runAndWait()
                break
            
            found_match = False

            # search AIML file for a matching pattern
            for category in aiml_root.findall('.//category'):
                pattern = category.find('pattern').text.lower()
                pattern = re.sub(r'[^\w\s]', '', pattern)
                pattern = " ".join([lemmatizer.lemmatize(word) for word in pattern.split()])
                similarity = cosine_similarity(vectorizer.fit_transform([user_input, pattern]))[0][1]
                if similarity > 0.8:
                    response = category.find('template').text
                    print("Bot:", response)
                    engine.say(response)
                    engine.runAndWait()
                    found_match = True
                    break

            # search CSV file for a matching question
            if not found_match:
                # load CSV file
                df = pd.read_csv('game_topic_qa.csv')
                df['question_lemmatized'] = df['Question'].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()]))
                df['similarity'] = df['question_lemmatized'].apply(lambda x: cosine_similarity(vectorizer.fit_transform([user_input, x]))[0][1])
                df = df.sort_values('similarity', ascending=False)
                df = df[df['similarity'] > 0.8]
                if not df.empty:
                    response = df.iloc[0]['Answer']
                    print("Bot:", response)
                    engine.say(response)
                    engine.runAndWait()
                    found_match = True

             # search Wikipedia for a matching article
                if not found_match:
                    try:
                        page = wikipedia.page(user_input)
                        response = page.content.split('\n')[0]
                        print("Bot:", response)
                        engine.say(response)
                        engine.runAndWait()
                        found_match = True
                    except wikipedia.exceptions.DisambiguationError as e:
                        response = "I'm sorry, I found multiple results for your query. Please try again with a more specific search term."
                        print("Bot:", response)
                        engine.say(response)
                        engine.runAndWait()
                    except wikipedia.exceptions.PageError as e:
                        pass

                # no match found in AIML, CSV, or Wikipedia
                if not found_match:
                    response = "I'm sorry, I didn't understand your question."
                    print("Bot:",response)
                    engine.say(response)
                    engine.runAndWait()

        back_or_exit = input("Press 'b' to go back to the main menu or 'e' to exit the program: ")
        if back_or_exit.lower() == 'b':
            continue
        else:
            break
##############################
#Part2
##############################        
    elif choice == "2":
        # Run program B if the user chooses option 2
        #  Initialise NLTK Inference
        from nltk.sem import Expression
        from nltk.inference import ResolutionProver
        import pandas
        import aiml
        import sys
        #  Initialise Knowledgebase. 
        read_expr = Expression.fromstring
        kb = []

        data = pandas.read_csv('kb.csv', header=None)
        [kb.append(read_expr(row)) for row in data[0]]

        # Check for inconsistencies in the knowledge base
        i = 0
        inconsistencies = False
        while i < len(kb):
            item = kb[i]
            if ResolutionProver().prove(item, kb, verbose=False):
                if ResolutionProver().prove(item.negate(), kb, verbose=False):
                    inconsistencies = True
                    break
            i += 1

        if inconsistencies:
            sys.exit("There are inconsistencies in the KB.")
        # Create a Kernel object. No string encoding (all I/O is unicode)
        kern = aiml.Kernel()
        kern.setTextEncoding(None)
        kern.bootstrap(learnFiles="mybot-logic.xml")
        # Welcome user
        print("Welcome to this chat bot. Please feel free to ask questions from me!")
        # Main loop
        while True:
            try:
                userInput = input("> ")
            except (KeyboardInterrupt, EOFError) as e:
                print("Bye!")
                break

            responseAgent = 'aiml'
            #activate selected response agent
            if responseAgent == 'aiml':
                answer = kern.respond(userInput)

            if answer[0] == '#':
                params = answer[1:].split('$')
                cmd = int(params[0])
                if cmd == 0:
                    print(params[1])
                    break
                elif cmd == 31:# if input pattern is "I know that * is *"
                    object,subject=params[1].split(' is ')
                    expr=read_expr(subject + '(' + object + ')')
                    if ResolutionProver().prove(expr.negate(),kb,verbose=False) is True:
                        print("This fact contradicts",subject)
                    else:
                        kb.append(expr) 
                        print('OK, I will remember that',object,'is', subject)
                elif cmd == 32:# if the input pattern is "check that * is *"
                    object,subject=params[1].split(' is ')
                    expr=read_expr(subject + '(' + object + ')')
                    if ResolutionProver().prove(expr,kb,verbose=False) is True:
                        print('Correct.')
                    elif ResolutionProver().prove(expr.negate(),kb,verbose=False) is True:
                        print("Incorrect")
                    else:
                        print("Sorry, I don't know") 
                elif cmd == 99:
                    print("I did not get that, please try again.")
            else:
                print(answer)
            
        back_or_exit = input("Press 'b' to go back to the main menu or 'e' to exit the program: ")
        if back_or_exit.lower() == 'b':
            continue
        else:
            break
##############################
#Part3
############################## 
    elif choice == "3":
        # Run program C if the user chooses option 3
        # Importing required libraries
        from tensorflow import keras
        import numpy as np
        import cv2
        import os
        import warnings
        
        # Disable TensorFlow warnings
        warnings.filterwarnings('ignore', category=FutureWarning)
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        warnings.filterwarnings('ignore', category=UserWarning)
        
        # Defining input and output parameters
        output_classes = 2 # number of classes
        input_shape = (150, 150, 3) # input image shape
        classes = ["lee_sin", "garen"]
        # Loading the model
        model = keras.models.load_model('game_classifier.h5')
        
        # Function to classify an image
        def classify_image(image_path):
            try:
                # Load and preprocess the image
                test_img = cv2.imread(image_path)
                if test_img is not None:
                    test_img = cv2.resize(test_img, (input_shape[0], input_shape[1]))
                    test_img = np.array(test_img).reshape(-1, input_shape[0], input_shape[1], input_shape[2])
                    test_img = test_img / 255.0
        
                    # Use the model to classify the image
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        prediction = model.predict(test_img)
                    class_idx = np.argmax(prediction)
                    class_name = classes[class_idx]
                    return class_name
                else:
                    return "Failed to load image."
            except:
                return "An error occurred while processing the image."
        
        # Running the program continuously until the user exits
        while True:
            # Getting the path to an image from the user
            image_path = input("Enter the path to an image (or type 'exit' to quit): ")
            if image_path.lower() == "exit":
                break
            
            # Classifying the image and displaying the result
            class_name = classify_image(image_path)
            print('The image contains:', class_name)
            
        back_or_exit = input("Press 'b' to go back to the main menu or 'e' to exit the program: ")
        if back_or_exit.lower() == 'b':
            continue
        else:
            break
    elif choice == "4":
        print("Bye!")
        # Exit the loop if user enters "exit"
        break
    else:
        print("Invalid input. Please choose a valid option.")

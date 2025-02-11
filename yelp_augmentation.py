from numpy.ma.core import negative
from openai import OpenAI
from tqdm import tqdm

# Instantiate OpenAI client
client = OpenAI(
    api_key='your-api-key'
)

sentiment = "negative" # or "positive"
n_sentences = 30000

# Define the prompt and parameters
positive_prompt = (
    """
    Generate diverse, concise, and natural positive sentences suitable for training a sentiment analysis model. 
    Each sentence should reflect a genuine and engaging tone, similar to those found in Yelp-style reviews. 
    Focus only on positive sentiments related to food, service, ambiance, or other enjoyable experiences. 
    The sentences should vary in length, intensity, and style, capturing different types of positivity and include informal review also. 
    Use the examples below as guidance for the tone and structure, particularly focusing on yelp positive examples:
    Avoid quotation marks, numbered lists, or bullet points. 
    Use the examples below as guidance for the tone and structure:
    
    Examples:
    The food was absolutely incredible and full of flavor
    The staff was so friendly and attentive to every detail
    I loved the cozy and welcoming atmosphere here
    The steak was cooked perfectly, and the wine pairing was excellent
    This place always exceeds my expectations
    The pizza was the best I’ve ever had, with a crispy crust and fresh toppings
    The desserts are to die for, especially the chocolate lava cake
    I can’t wait to come back again for another amazing experience
    The coffee is rich, smooth, and brewed to perfection
    Their attention to detail in every dish is outstanding
    
    Yelp positive examples:
    excellent food .
    superb customer service .
    they also have daily specials and ice cream which is really good .
    it 's a good toasted hoagie .
    the staff is friendly .
    good bar food .
    good service .
    soup of day is homemade and lots of specials .
    great place for lunch or bar snacks and beer .
    the new range looks amazing .
    this place was very good .
    but the people are friendly & the food is good .
    traditional `` mom 'n pop '' quality and perfection .
    the best fish and chips you 'll ever enjoy and equally superb fried shrimp .
    you will love it .
    wonderful reuben .
    good fish sandwich .
    this is a hidden gem , no really .
    it took us forever to find but well worth it .
    huge sandwich !
    i added mushrooms , it was very flavorful .
    my boyfriend got the fish sandwich , he enjoyed it as well .
    fast and friendly service .
    will definitely be back .
    my dad 's favorite , as he knows the original owners .
    huge burgers , fish sandwiches , salads .
    decent service .
    love my hometown favorites .
    ample free parking makes it easy to go there .
    i love everything about this place .
    it takes decades of good food and good vibes .
    also , the food is really good .
    excellent fish sandwich , wonderful reuben sandwich , even the stuffed cabbage tastes homemade .
    is a delightful hostess and makes you feel welcome .
    restaurant personnel are pleasant .
    great food & service every visit .
    they do so many things right there especially taking care of the customers .
    last week we had the wonderful fish - it has always been good .
    it was truly outstanding !
    the homemade tomato soup with asiago cheese was delicious as well .
    thanks chef tony !
    love , love !
    it 's always very well maintained .
    the carts are in excellent shape , all electric and all equipped with gps .
    challenging but fun course !
    beautiful views and lots of variety of length and layout of holes .
    i 'll definitely be back !
    the service and prices were great.
    i had the buffalo chicken sandwich and it was delicious .
    a cool bar off the beaten path that is a worth a trip .
    awesome drink specials during happy hour .
    fantastic wings that are crispy and delicious , wing night on tuesday and thursday !
    the sandwiches are always amazing just as i remember .
    the staff is amazing and friendly .
    great place for lunch as well .
    friendly staff , good food , great beer selection , and relaxing atmosphere .
    great wings and the buffalo chicken pizza is the best i 've had .
    the sandwiches are all on thick cut italian bread and fresh .
    if we ever get to the pittsburgh area again , we will go back !
    perfect joint !
    it 's a nice little neighborhood .
    was in town on biz and my hotel was right up the street .
    fantastic !
    great food , excellent beer selection !
    bartenders are great !
    it is a great hometown neighborhood bar with good people and friendly staff .
    excellent wings !
    i go here to get away and enjoy a nice drink and great company .
    the service is always top notch and customer service is awesome .
    first class management and food .
    it looks a lot better than it did in previous years .
    awesome experience !
    i took my wife 's car in for new tires on good friday .
    the candy hut is fun to spend your spare change .
    it ended up working very well and they were very polite .
    nice , suppose to be haunted .
    awesome historic building high on top of the hill in carnegie .
    plenty of free parking and at only _num_ dollars its a true bargain .
    i heart king 's .
    staff always friendly and the deli staff in both locations are great !
    shopping these two stores is as pleasant as it gets .
    nice friendly staff .
    great layout .
    and again , so clean !
    this place will give you the best fish sandwich in pittsburgh .
    enjoy ... .
    i highly recommend .
    great bar .
    good place to watch a game .
    great local bar .
    theresa does a great job .
    sandwiches are excellent .
    the food is basic and simple , but it is delicious .
    sometimes simple is better ; this is definitely one of those times .
    generous portions of really great food .
    best fish sandwiches .
    good luck getting a seat , that 's all i have to say .
    diner food is what 's up and i like it .
    stick to basics and this is the best place in or around the 'burgh .
    okay , let 's first set expectations .
    
    Generate sentences with a mix of contexts (food, service, ambiance), a range of sentiment intensities (mild to highly enthusiastic), and varying lengths. 
    Avoid repetition and ensure each sentence feels unique and authentic.
    You must generate a total of 10 sentences. The average length of each sentence should be around 10 words.
    """
)

negative_prompt = (
    """
    Generate diverse, concise, and natural negative sentences suitable for training a sentiment analysis model. 
    Each sentence should reflect a genuine and engaging tone, similar to those found in Yelp-style reviews. 
    Focus only on negative sentiments related to food, service, ambiance, or other disappointing experiences. 
    The sentences should vary in length, intensity, and style, capturing different types of negativity and include informal review styles as well.
    
    Avoid quotation marks, numbered lists, or bullet points. Use the examples below as guidance for the tone and structure:
    
    Examples:
    The service was extremely slow, and our food arrived cold
    The coffee tasted burnt and was undrinkable
    The ambiance was too noisy and ruined the dining experience
    The steak was overcooked and lacked seasoning
    I had high hopes, but the food was a huge letdown
    The portions were ridiculously small for the price they charge
    The staff seemed uninterested and made us feel unwelcome
    The restaurant smelled odd, and the tables were sticky
    The dessert was stale and definitely not fresh
    The salad was soggy and tasted like it came from a bag
    
    Yelp negative examples:
    i was sadly mistaken .
    so on to the hoagies , the italian is general run of the mill .
    minimal meat and a ton of shredded lettuce .
    nothing really special & not worthy of the $ _num_ price tag .
    second , the steak hoagie , it is atrocious .
    i had to pay $ _num_ to add cheese to the hoagie .
    she told me there was a charge for the dressing on the side .
    are you kidding me ?
    i was not going to pay for the dressing on the side .
    i ordered it without lettuce , tomato , onions , or dressing .
    are you kidding me ?
    i paid $ _num_ to add sauted mushrooms , onions , and cheese .
    in this case , never .
    ( the hoagie bun was better than average . )
    wake up or you are going to lose your business .
    this place has none of them .
    it is april and there are no grass tees yet .
    there is no grass on the range .
    bottom line , this place sucks .
    someone should buy this place and turn it into what it should be .
    very disappointed in the customer service .
    we will not be back .
    the iced tea is also terrible tasting .
    used to go there for tires , brakes , etc .
    plus , _num_ of the new tires went flat within _num_ weeks .
    terrible .
    i was originally told it would take _num_ mins .
    slow , over priced , i 'll go elsewhere next time .
    so frustrated .
    never going back .
    they seem overpriced , too .
    do n't waste your time .
    not a call , not the estimate , nothing .
    shortly after returning home she started to cough and gag like .
    she was very uncomfortable .
    at this point they were open and would be for another hour .
    wait _num_ hrs .
    she has to wait _num_ days to be seen .
    absolutely the worst care in all my experience with vets !
    this place went from great to horrible .
    absolutely no problems .
    noisy snow tires in the summer !
    i told him they are too noisy .
    very rude and once they get paid have no ethics .
    i dropped off _num_ shirts at this establishment just before thanksgiving .
    they have offered $ _num_ a shirt for the shirts they lost .
    i would n't use their services again if you paid me to .
    sign is blank and all fixtures from the store have been removed .
    bread or not , it still was n't great .
    it was a basic salad , no big deal .
    i never imagined how confusing a question that could possibly be .
    nah ... why bother when there are so many other options around .
    just do n't go hungry if you actually have tastebuds .
    but apparently , mama has left the kitchen .
    as for my mushroom and cheese omelette , the cheese was lacking .
    if i wanted a microwave omelet i would have gone to burger king .
    two scrambled whole eggs , italian toast , and home fries .
    in a considerably nasty tone .
    trash .
    guess who was n't too busy , carnegie coffee company down the street .
    nothing special .
    it was hot in there and i do n't think they have ac .
    it does n't say cash only anywhere inside or on the menu either .
    there is an atm , but it has a $ _num_ charge to use .
    ca n't believe a place in _num_ does n't take cards .
    the coffee at this place was brewed to perfection which a appreciate .
    i fed my whole family breakfast ( _num_ of us ) for $ _num_ .
    not to mention my allergies were in full effect .
    the bathroom was so inconvenient to get in and out of .
    breakfast was pathetic .
    i 'll be sure to spend my money elsewhere next time .
    if i could give this place less than one star , i would .
    apparently , we were out of luck .
    it 's a tiny long bar just like the old irish pubs .
    no more smoking allowed .
    same lousy service .
    i really really want this place to do better .
    we need more irish pubs and local old time watering holes .
    whatever the case , they have cold guinness .
    some of the worst pizza i 've ever had .
    we used a coupon from the paper for a _num_ topping _num_ cut sicilian .
    the onions were huge chunks and the mushrooms were straight from a can .
    it was gross .
    the veggies were old and wilted , and there was no dressing on either .
    no flavor or seasoning and the texture was reminiscent of spam .
    i have ordered from here in the past and always been disappointed .
    what a mistake .
    the business is not run the way it has been in the past .
    this is probably the worst business in carnegie .
    is it possible to give zero stars ?
    the inside is very dated and appears to be poorly cared for .
    it smelled like burnt fish when we walked in the door .
    the fish was old and cold and fries were a very soft .
    the corn lacked butter .
    service was n't too bad - nice people .
    unfortunately , there are only two reviews .
    they did not acknowledge me .
    the door i was at had signage ( open hours ) .
    take it from me ; avoid this place at all cost .
    does n't even taste good as american style chinese food .
    
    Generate sentences with a mix of contexts (food, service, ambiance), a range of sentiment intensities (mild disappointment to strong dissatisfaction), and varying lengths. 
    Avoid repetition and ensure each sentence feels unique and authentic.
    You must generate a total of 10 sentences. The average length of each sentence should be around 10 words.
    """
)

# Set the prompt and output file based on the sentiment
if sentiment == "positive":
    prompt = positive_prompt
    output_file = "data/yelp/train_augmented.1.txt"
else:
    prompt = negative_prompt
    output_file = "data/yelp/train_augmented.0.txt"

# Generate sentences using the prompt
with open(output_file, "w") as f:
    for i in tqdm(range(n_sentences // 10)):
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Ensure the correct model name for GPT-4-mini
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}],
            n=1,
            temperature=0.7
        )
        # Extract the generated sentences from the response
        answer = response.choices[0].message.content
        sentences = answer.split("\n")
        for sentence in sentences:
            f.write(sentence + "\n")

# Print the completion message
print(f"Sentences generated and saved to {output_file}.")
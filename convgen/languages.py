"""Language + region table for multilingual conversation generation.

Each entry ties a language to a real, representative speaker region with a
handful of concrete grounding facts (climate, staple crops, notable
places/cities, currency, financial regulator) — used only by the domains
where region genuinely matters (see domains.py's REGION_SENSITIVE set).
Facts are deliberately few and concrete rather than exhaustive; the point is
real grounding, not an encyclopedia entry.

`instruction` is the meta-instruction embedded in every persona system
prompt for this language: the system prompt itself is authored in English
(so the underlying model reliably understands its persona/task), but
explicitly directs the model to write the actual conversation in the target
language and script. This is the standard, realistic way to drive
multilingual generation from models that were not hand-authored natively
per language.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Language:
    slug: str
    name: str
    native_name: str
    region: str
    climate: str
    staples: list[str]  # crops/produce, used by Agriculture
    places: list[str]  # districts/cities, used by region-sensitive domains
    currency: str
    regulator: str  # central bank / financial regulator, used by Banking/Insurance/Finance
    person_names: list[str]  # common first names, for persona naming

    @property
    def instruction(self) -> str:
        return (
            f"Write your entire reply in {self.name} ({self.native_name}), using its "
            f"native script. Do not switch to English or transliterate — write natively "
            f"in {self.name} the way a fluent speaker from {self.region} actually would."
        )


LANGUAGES: list[Language] = [
    Language("arabic", "Arabic", "العربية", "the Gulf and North Africa", "hot arid",
             ["dates", "wheat", "olives"], ["Cairo", "Jeddah", "Dubai", "Amman", "Tunis"],
             "the local currency (e.g. EGP/AED/SAR as appropriate)", "the national central bank",
             ["Ahmed", "Fatima", "Youssef", "Layla", "Omar", "Mariam"]),
    Language("assamese", "Assamese", "অসমীয়া", "Assam, India", "subtropical monsoon, Brahmaputra floods",
             ["tea", "rice", "jute"], ["Guwahati", "Jorhat", "Dibrugarh", "Silchar", "Tezpur"],
             "Indian Rupee (INR)", "the Reserve Bank of India",
             ["Bhaskar", "Rituporna", "Dipankar", "Mridusmita", "Hemanta", "Junmoni"]),
    Language("baloch", "Balochi", "بلوچی", "Balochistan (Pakistan/Iran border region)", "arid, mountainous",
             ["dates", "wheat", "sorghum"], ["Quetta", "Turbat", "Gwadar", "Khuzdar"],
             "Pakistani Rupee (PKR)", "the State Bank of Pakistan",
             ["Dilmurad", "Sammi", "Mir Gul", "Bibi Gul", "Nasir", "Zarina"]),
    Language("bengali", "Bengali", "বাংলা", "West Bengal, India and Bangladesh", "tropical monsoon, June-September",
             ["rice", "jute", "tea"], ["Kolkata", "Nadia", "Murshidabad", "Dhaka", "Rajshahi"],
             "Indian Rupee or Bangladeshi Taka as appropriate", "the Reserve Bank of India or Bangladesh Bank",
             ["Arup", "Priyanka", "Debashish", "Ananya", "Shubho", "Mousumi"]),
    Language("bishnupriya_manipuri", "Bishnupriya Manipuri", "বিষ্ণুপ্রিয়া মণিপুরী", "Manipur and Assam, India",
             "humid subtropical, hilly", ["rice", "areca nut"], ["Imphal", "Cachar", "Sylhet-border villages"],
             "Indian Rupee (INR)", "the Reserve Bank of India",
             ["Bijoy", "Rina", "Sanjit", "Lata", "Bipin", "Nomita"]),
    Language("bodo", "Bodo", "बड़ो", "Bodoland, Assam, India", "subtropical monsoon",
             ["rice", "areca nut", "mustard"], ["Kokrajhar", "Udalguri", "Chirang", "Baksa"],
             "Indian Rupee (INR)", "the Reserve Bank of India",
             ["Rwmwswm", "Mwnsri", "Dilip", "Anjali", "Gautam", "Sunita"]),
    Language("burmese", "Burmese", "မြန်မာဘာသာ", "Myanmar", "tropical monsoon, Irrawaddy delta",
             ["rice", "pulses", "sesame"], ["Yangon", "Mandalay", "Bago", "Naypyidaw"],
             "Myanmar Kyat (MMK)", "the Central Bank of Myanmar",
             ["Aung", "Su Su", "Htet", "Moe Moe", "Zaw", "Nilar"]),
    Language("dari", "Dari (Afghan Persian)", "دری", "Kabul and central Afghanistan", "arid, mountainous",
             ["wheat", "pomegranates", "almonds"], ["Kabul", "Herat", "Mazar-i-Sharif", "Bamyan"],
             "Afghani (AFN)", "Da Afghanistan Bank",
             ["Ahmad", "Zarghuna", "Wahid", "Freshta", "Naseer", "Soraya"]),
    Language("dogri", "Dogri", "डोगरी", "Jammu region, India", "sub-humid, hilly terraced farming",
             ["maize", "wheat", "basmati rice"], ["Jammu", "Kathua", "Udhampur", "Samba"],
             "Indian Rupee (INR)", "the Reserve Bank of India",
             ["Vikram", "Poonam", "Rakesh", "Kiran", "Deepak", "Sunita"]),
    Language("dzongkha", "Dzongkha", "རྫོང་ཁ", "Bhutan", "temperate Himalayan, valley rice terraces",
             ["rice", "maize", "apples"], ["Thimphu", "Paro", "Punakha", "Wangdue"],
             "Bhutanese Ngultrum (BTN)", "the Royal Monetary Authority of Bhutan",
             ["Tenzin", "Pema", "Karma", "Dechen", "Sonam", "Yeshi"]),
    Language("english", "English", "English", "a generalized English-speaking context", "temperate",
             ["wheat", "corn", "mixed produce"], ["a mid-sized city", "a regional town"],
             "US Dollar or local currency as appropriate", "the relevant national regulator",
             ["Alex", "Jordan", "Sam", "Taylor", "Morgan", "Casey"]),
    Language("french", "French", "Français", "France", "temperate oceanic",
             ["wheat", "grapes", "sunflowers"], ["Lyon", "Toulouse", "Rennes", "Dijon"],
             "Euro (EUR)", "the Banque de France / ACPR",
             ["Antoine", "Camille", "Julien", "Manon", "Nicolas", "Chloé"]),
    Language("gujarati", "Gujarati", "ગુજરાતી", "Gujarat, India", "semi-arid, coastal",
             ["cotton", "groundnut", "cumin"], ["Ahmedabad", "Rajkot", "Surat", "Bhavnagar"],
             "Indian Rupee (INR)", "the Reserve Bank of India",
             ["Kiran", "Foram", "Jignesh", "Riddhi", "Mehul", "Nisha"]),
    Language("hindi", "Hindi", "हिन्दी", "the Gangetic plains, North India", "subtropical monsoon",
             ["wheat", "sugarcane", "rice"], ["Lucknow", "Patna", "Kanpur", "Varanasi"],
             "Indian Rupee (INR)", "the Reserve Bank of India",
             ["Rohit", "Anjali", "Sandeep", "Pooja", "Manoj", "Rekha"]),
    Language("indonesian", "Indonesian", "Bahasa Indonesia", "Indonesia", "tropical, rice paddies",
             ["rice", "palm oil", "cassava"], ["Yogyakarta", "Surabaya", "Bandung", "Medan"],
             "Indonesian Rupiah (IDR)", "Bank Indonesia / OJK",
             ["Budi", "Siti", "Andi", "Dewi", "Agus", "Putri"]),
    Language("kannada", "Kannada", "ಕನ್ನಡ", "Karnataka, India", "semi-arid Deccan plateau",
             ["coffee", "ragi", "areca nut"], ["Mysuru", "Hubballi", "Belagavi", "Mangaluru"],
             "Indian Rupee (INR)", "the Reserve Bank of India",
             ["Manjunath", "Lakshmi", "Prakash", "Sowmya", "Ravi", "Kavya"]),
    Language("kashmiri", "Kashmiri", "کٲشُر", "the Kashmir Valley", "temperate to cold, apple orchards",
             ["apples", "saffron", "walnuts"], ["Srinagar", "Anantnag", "Baramulla", "Pulwama"],
             "Indian Rupee (INR)", "the Reserve Bank of India",
             ["Bilal", "Sameera", "Tariq", "Nusrat", "Aamir", "Rukhsana"]),
    Language("kazakh", "Kazakh", "Қазақша", "Kazakhstan", "continental steppe",
             ["wheat", "barley", "millet"], ["Almaty", "Shymkent", "Karaganda", "Astana"],
             "Kazakhstani Tenge (KZT)", "the National Bank of Kazakhstan",
             ["Nurlan", "Aigerim", "Serik", "Dana", "Yerlan", "Gulnaz"]),
    Language("konkani", "Konkani", "कोंकणी", "Goa and coastal Karnataka/Maharashtra", "tropical monsoon, coastal",
             ["coconut", "cashew", "fish"], ["Panaji", "Margao", "Mangaluru", "Ratnagiri"],
             "Indian Rupee (INR)", "the Reserve Bank of India",
             ["Sameer", "Priya", "Rohan", "Neha", "Vinod", "Shalini"]),
    Language("maithili", "Maithili", "मैथिली", "the Mithila region (Bihar, India / Nepal terai)",
             "subtropical, flood-prone Kosi basin", ["rice", "maize", "makhana (fox nut)"],
             ["Darbhanga", "Madhubani", "Janakpur", "Muzaffarpur"], "Indian Rupee or Nepali Rupee as appropriate",
             "the Reserve Bank of India or Nepal Rastra Bank",
             ["Anand", "Nutan", "Shailesh", "Rekha", "Bipin", "Sunaina"]),
    Language("malayalam", "Malayalam", "മലയാളം", "Kerala, India", "tropical monsoon, backwaters",
             ["pepper", "cardamom", "coconut", "rubber"], ["Kochi", "Kozhikode", "Thrissur", "Alappuzha"],
             "Indian Rupee (INR)", "the Reserve Bank of India",
             ["Arjun", "Anjali", "Vishnu", "Divya", "Sanjay", "Meera"]),
    Language("marathi", "Marathi", "मराठी", "Maharashtra, India", "Deccan plateau, drought-prone interior",
             ["cotton", "sugarcane", "onions"], ["Pune", "Nagpur", "Nashik", "Aurangabad"],
             "Indian Rupee (INR)", "the Reserve Bank of India",
             ["Rahul", "Sneha", "Vikas", "Priyanka", "Sachin", "Vaishali"]),
    Language("meitei", "Meitei (Manipuri)", "মৈতৈলোন্", "the Imphal Valley, Manipur, India",
             "humid subtropical, hilly", ["rice", "vegetables"], ["Imphal", "Bishnupur", "Thoubal"],
             "Indian Rupee (INR)", "the Reserve Bank of India",
             ["Rajkumar", "Ibemhal", "Naoba", "Chaoba", "Tomba", "Ranjita"]),
    Language("nepali", "Nepali", "नेपाली", "Nepal", "Himalayan, terraced hill farming",
             ["rice", "maize", "millet"], ["Kathmandu", "Pokhara", "Biratnagar", "Dharan"],
             "Nepali Rupee (NPR)", "Nepal Rastra Bank",
             ["Bikash", "Sabina", "Prakash", "Sunita", "Ramesh", "Anita"]),
    Language("odia", "Odia", "ଓଡ଼ିଆ", "Odisha, India", "tropical, cyclone-prone coast, Mahanadi delta",
             ["rice", "vegetables"], ["Bhubaneswar", "Cuttack", "Puri", "Balasore"],
             "Indian Rupee (INR)", "the Reserve Bank of India",
             ["Bibhuti", "Snigdha", "Debasis", "Rashmita", "Subrat", "Ipsita"]),
    Language("pashto", "Pashto", "پښتو", "the Pashtun belt (Afghanistan / Pakistan KPK)", "arid, mountainous",
             ["wheat", "fruit orchards"], ["Peshawar", "Kandahar", "Jalalabad", "Quetta"],
             "Pakistani Rupee or Afghani as appropriate", "the State Bank of Pakistan or Da Afghanistan Bank",
             ["Rahim", "Spogmai", "Zahid", "Laila", "Yousaf", "Gulalai"]),
    Language("portuguese", "Portuguese", "Português", "Brazil", "tropical to subtropical",
             ["soy", "coffee", "sugarcane"], ["Belo Horizonte", "Curitiba", "Recife", "Porto Alegre"],
             "Brazilian Real (BRL)", "the Banco Central do Brasil",
             ["Lucas", "Fernanda", "Rafael", "Juliana", "Gustavo", "Camila"]),
    Language("punjabi", "Punjabi", "ਪੰਜਾਬੀ", "Punjab (India/Pakistan)", "subtropical, canal-irrigated plains",
             ["wheat", "rice", "cotton"], ["Amritsar", "Ludhiana", "Lahore", "Jalandhar"],
             "Indian Rupee or Pakistani Rupee as appropriate", "the Reserve Bank of India or State Bank of Pakistan",
             ["Harpreet", "Simran", "Gurpreet", "Manpreet", "Jaspal", "Navjot"]),
    Language("sanskrit", "Sanskrit", "संस्कृतम्", "a classical North Indian context", "subtropical",
             ["rice", "wheat"], ["Varanasi", "Pune", "Mysuru", "Tirupati"], "Indian Rupee (INR)",
             "the Reserve Bank of India",
             ["Arjun", "Sita", "Krishna", "Radha", "Rama", "Lakshmi"]),
    Language("santhali", "Santhali", "ᱥᱟᱱᱛᱟᱲᱤ", "the Santhal Parganas (Jharkhand/West Bengal/Odisha)",
             "subtropical, forest-adjacent", ["rice", "maize", "millet"],
             ["Dumka", "Jamtara", "Pakur", "Purulia"], "Indian Rupee (INR)", "the Reserve Bank of India",
             ["Sido", "Phulo", "Chunkad", "Mangri", "Baskey", "Hembrom"]),
    Language("simplified_mandarin", "Simplified Mandarin", "简体中文", "China", "varies north to south",
             ["rice (south)", "wheat (north)", "vegetables"], ["Chengdu", "Wuhan", "Xi'an", "Hangzhou"],
             "Renminbi (CNY)", "the People's Bank of China",
             ["Wei", "Li Na", "Jun", "Xiaoyu", "Hao", "Mei"]),
    Language("sindhi", "Sindhi", "سنڌي", "Sindh, Pakistan", "arid, Indus-irrigated",
             ["wheat", "cotton", "sugarcane"], ["Karachi", "Hyderabad", "Sukkur", "Larkana"],
             "Pakistani Rupee (PKR)", "the State Bank of Pakistan",
             ["Ashfaq", "Sassui", "Rasheed", "Marvi", "Ayaz", "Naseem"]),
    Language("sinhala", "Sinhala", "සිංහල", "Sri Lanka", "tropical monsoon",
             ["tea", "rice", "rubber"], ["Kandy", "Galle", "Kurunegala", "Anuradhapura"],
             "Sri Lankan Rupee (LKR)", "the Central Bank of Sri Lanka",
             ["Nimal", "Kumari", "Sunil", "Chamari", "Ruwan", "Dilani"]),
    Language("spanish", "Spanish", "Español", "Mexico", "varies, from arid north to tropical south",
             ["maize", "agave", "coffee"], ["Guadalajara", "Puebla", "Monterrey", "Mérida"],
             "Mexican Peso (MXN)", "Banco de México / CNBV",
             ["Diego", "Valentina", "Javier", "Camila", "Andrés", "Lucía"]),
    Language("swahili", "Swahili", "Kiswahili", "Kenya and Tanzania", "tropical savanna",
             ["maize", "coffee", "tea"], ["Nairobi", "Mombasa", "Dodoma", "Arusha"],
             "Kenyan Shilling or Tanzanian Shilling as appropriate", "the Central Bank of Kenya or Bank of Tanzania",
             ["Juma", "Amina", "Baraka", "Zawadi", "Hassan", "Neema"]),
    Language("tamil", "Tamil", "தமிழ்", "Tamil Nadu, India", "semi-arid interior, monsoon coast",
             ["rice (Cauvery delta)", "sugarcane", "cotton"], ["Madurai", "Coimbatore", "Tiruchirappalli", "Salem"],
             "Indian Rupee (INR)", "the Reserve Bank of India",
             ["Karthik", "Priya", "Suresh", "Divya", "Ganesh", "Kavitha"]),
    Language("telugu", "Telugu", "తెలుగు", "Andhra Pradesh and Telangana, India",
             "tropical, Krishna-Godavari delta", ["rice", "chilies", "cotton"],
             ["Vijayawada", "Warangal", "Guntur", "Nellore"], "Indian Rupee (INR)", "the Reserve Bank of India",
             ["Venkatesh", "Lakshmi", "Srinivas", "Padma", "Ravi", "Swathi"]),
    Language("thai", "Thai", "ภาษาไทย", "Thailand", "tropical monsoon",
             ["jasmine rice", "sugarcane", "rubber"], ["Chiang Mai", "Khon Kaen", "Nakhon Ratchasima", "Ayutthaya"],
             "Thai Baht (THB)", "the Bank of Thailand",
             ["Somchai", "Suda", "Anan", "Nari", "Preecha", "Malee"]),
    Language("tibetan", "Tibetan", "བོད་སྐད", "the Tibetan plateau", "cold arid, high-altitude",
             ["barley", "potatoes", "buckwheat"], ["Lhasa", "Shigatse", "Nagqu"], "Renminbi (CNY)",
             "the People's Bank of China",
             ["Tenzin", "Dolma", "Tashi", "Yangchen", "Norbu", "Lhamo"]),
    Language("urdu", "Urdu", "اردو", "Punjab and North India / Pakistan", "subtropical plains",
             ["wheat", "rice", "sugarcane"], ["Lahore", "Karachi", "Lucknow", "Hyderabad (Deccan)"],
             "Pakistani Rupee or Indian Rupee as appropriate", "the State Bank of Pakistan or Reserve Bank of India",
             ["Imran", "Ayesha", "Faisal", "Sana", "Adeel", "Nadia"]),
    Language("uzbek", "Uzbek", "Oʻzbekcha", "Uzbekistan", "continental arid, irrigated",
             ["cotton", "wheat", "melons"], ["Samarkand", "Bukhara", "Namangan", "Andijan"],
             "Uzbekistani Som (UZS)", "the Central Bank of Uzbekistan",
             ["Bobur", "Nodira", "Jasur", "Madina", "Sardor", "Gulnora"]),
    Language("vietnamese", "Vietnamese", "Tiếng Việt", "Vietnam", "tropical monsoon, Mekong and Red River deltas",
             ["rice", "coffee", "rubber"], ["Da Nang", "Can Tho", "Hue", "Hai Phong"],
             "Vietnamese Dong (VND)", "the State Bank of Vietnam",
             ["Minh", "Linh", "Tuan", "Huong", "Duc", "Mai"]),
]

BY_SLUG: dict[str, Language] = {lang.slug: lang for lang in LANGUAGES}

assert len(LANGUAGES) == 42, f"expected 42 languages, got {len(LANGUAGES)}"

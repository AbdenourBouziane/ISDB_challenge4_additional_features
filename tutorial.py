import os
import streamlit as st
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from googletrans import Translator

# Load environment variables
load_dotenv()

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]

# Initialize translator
translator = Translator()

# Define AAOIFI standards dictionary (simplified versions)
standards = {
    "FAS 4": {
        "title_en": "Currency Translation",
        "title_ar": "ترجمة العملات",
        "description_en": "Standard for recording foreign currency transactions and translating financial statements.",
        "description_ar": "معيار لتسجيل معاملات العملات الأجنبية وترجمة البيانات المالية."
    },
    "FAS 7": {
        "title_en": "Investments in Real Estate",
        "title_ar": "الاستثمارات العقارية",
        "description_en": "Standard for accounting for investments in properties for rental or capital appreciation.",
        "description_ar": "معيار محاسبة الاستثمارات في العقارات للتأجير أو زيادة رأس المال."
    },
    "FAS 10": {
        "title_en": "Istisna'a and Parallel Istisna'a",
        "title_ar": "الاستصناع والاستصناع الموازي",
        "description_en": "Standard for manufacturing contracts where payment is made in installments.",
        "description_ar": "معيار لعقود التصنيع حيث يتم الدفع على أقساط."
    },
    "FAS 28": {
        "title_en": "Ijarah and Ijarah Muntahia Bittamleek",
        "title_ar": "الإجارة والإجارة المنتهية بالتمليك",
        "description_en": "Standard for lease agreements and leases ending with ownership transfer.",
        "description_ar": "معيار لاتفاقيات الإيجار والإيجارات المنتهية بنقل الملكية."
    },
    "FAS 32": {
        "title_en": "Investment Agency (Al-Wakala Bi Al-Istithmar)",
        "title_ar": "وكالة الاستثمار (الوكالة بالاستثمار)",
        "description_en": "Standard for investment agency relationships between investors and agents.",
        "description_ar": "معيار لعلاقات وكالة الاستثمار بين المستثمرين والوكلاء."
    }
}

# Example use cases for each standard
examples = {
    "FAS 4": {
        "title_en": "Converting USD to EUR for international trade",
        "title_ar": "تحويل الدولار الأمريكي إلى اليورو للتجارة الدولية",
        "scenario_en": """
        Al Baraka Bank needs to record a transaction where they purchased machinery from a European supplier:
        - Purchase price: €500,000
        - Exchange rate on purchase date: 1 EUR = 1.10 USD
        - Exchange rate on payment date (30 days later): 1 EUR = 1.12 USD
        How should this be recorded in the books?
        """,
        "scenario_ar": """
        يحتاج بنك البركة إلى تسجيل معاملة شراء آلات من مورد أوروبي:
        - سعر الشراء: 500,000 يورو
        - سعر الصرف في تاريخ الشراء: 1 يورو = 1.10 دولار أمريكي
        - سعر الصرف في تاريخ الدفع (بعد 30 يومًا): 1 يورو = 1.12 دولار أمريكي
        كيف يجب تسجيل هذا في الدفاتر؟
        """
    },
    "FAS 7": {
        "title_en": "Acquiring an office building for rental",
        "title_ar": "الاستحواذ على مبنى مكتبي للتأجير",
        "scenario_en": """
        Islamic Finance House purchased a commercial building:
        - Purchase price: $5,000,000
        - Legal fees: $50,000
        - Building improvements: $200,000
        - Expected rental income: $400,000 per year
        How should this investment be recorded and measured?
        """,
        "scenario_ar": """
        اشترى بيت التمويل الإسلامي مبنى تجاري:
        - سعر الشراء: 5,000,000 دولار
        - الرسوم القانونية: 50,000 دولار
        - تحسينات المبنى: 200,000 دولار
        - الدخل المتوقع من الإيجار: 400,000 دولار سنويًا
        كيف يجب تسجيل وقياس هذا الاستثمار؟
        """
    },
    "FAS 10": {
        "title_en": "Manufacturing contract for custom equipment",
        "title_ar": "عقد تصنيع لمعدات مخصصة",
        "scenario_en": """
        Al Salam Bank entered into an Istisna'a contract with a manufacturer:
        - Contract value: $1,000,000 for custom manufacturing equipment
        - Payment schedule: 30% upfront, 30% halfway, 40% upon delivery
        - Manufacturing period: 8 months
        How should the bank record this transaction?
        """,
        "scenario_ar": """
        دخل بنك السلام في عقد استصناع مع مصنّع:
        - قيمة العقد: 1,000,000 دولار لمعدات تصنيع مخصصة
        - جدول الدفع: 30٪ مقدمًا، 30٪ في المنتصف، 40٪ عند التسليم
        - فترة التصنيع: 8 أشهر
        كيف يجب على البنك تسجيل هذه المعاملة؟
        """
    },
    "FAS 28": {
        "title_en": "Leasing equipment with ownership transfer",
        "title_ar": "تأجير معدات مع نقل الملكية",
        "scenario_en": """
        Alpha Islamic Bank entered into an Ijarah MBT for a generator:
        - Generator cost: $450,000
        - Import tax and freight: $42,000
        - Lease term: 2 years
        - Annual rental: $300,000
        - Purchase option at end: $3,000
        How should Alpha Bank record this transaction?
        """,
        "scenario_ar": """
        دخل بنك ألفا الإسلامي في إجارة منتهية بالتمليك لمولد كهربائي:
        - تكلفة المولد: 450,000 دولار
        - ضريبة الاستيراد والشحن: 42,000 دولار
        - مدة الإيجار: سنتان
        - الإيجار السنوي: 300,000 دولار
        - خيار الشراء في النهاية: 3,000 دولار
        كيف يجب على بنك ألفا تسجيل هذه المعاملة؟
        """
    },
    "FAS 32": {
        "title_en": "Investment agency relationship",
        "title_ar": "علاقة وكالة استثمار",
        "scenario_en": """
        Qatar Islamic Bank accepted $10M from investors on Wakala basis:
        - Expected profit rate: 5% annually
        - Bank's agency fee: 20% of profits above 5%
        - Investment term: 1 year
        - Actual return achieved: 7%
        How should this be accounted for?
        """,
        "scenario_ar": """
        قبل بنك قطر الإسلامي 10 مليون دولار من المستثمرين على أساس الوكالة:
        - معدل الربح المتوقع: 5٪ سنويًا
        - رسوم وكالة البنك: 20٪ من الأرباح فوق 5٪
        - مدة الاستثمار: سنة واحدة
        - العائد الفعلي المحقق: 7٪
        كيف يجب المحاسبة عن ذلك؟
        """
    }
}

class IslamicFinanceStandardsExplainer:
    def __init__(self):
        # Initialize language model
        self.chat_model = ChatOpenAI(model_name="gpt-4", temperature=0.5)
        
        # Initialize explanation chain with English system message
        self.explanation_template_en = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "You are an expert in Islamic Finance standards, particularly AAOIFI standards. "
                "Explain the accounting treatment for the given scenario in simple terms that a non-specialist can understand. "
                "Use step-by-step explanations and include the journal entries where appropriate."
            ),
            HumanMessagePromptTemplate.from_template(
                "Standard: {standard_title}\n\n"
                "Scenario: {scenario}\n\n"
                "Please explain:\n"
                "1. What this standard is about\n"
                "2. How to account for this transaction step-by-step\n"
                "3. The proper journal entries\n"
                "4. Why this method complies with Islamic finance principles"
            )
        ])
        
        # Initialize explanation chain with Arabic system message
        self.explanation_template_ar = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "أنت خبير في معايير التمويل الإسلامي، خاصة معايير هيئة المحاسبة والمراجعة للمؤسسات المالية الإسلامية. "
                "قم بشرح المعالجة المحاسبية للسيناريو المعطى بمصطلحات بسيطة يمكن لغير المتخصص فهمها. "
                "استخدم شرحًا خطوة بخطوة وقم بتضمين قيود اليومية حيثما كان ذلك مناسبًا."
            ),
            HumanMessagePromptTemplate.from_template(
                "المعيار: {standard_title}\n\n"
                "السيناريو: {scenario}\n\n"
                "يرجى شرح:\n"
                "1. ما هو هذا المعيار\n"
                "2. كيفية المحاسبة عن هذه المعاملة خطوة بخطوة\n"
                "3. قيود اليومية المناسبة\n"
                "4. لماذا تتوافق هذه الطريقة مع مبادئ التمويل الإسلامي"
            )
        ])
        
        # Initialize feedback chain
        self.feedback_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "You are an expert in Islamic Finance standards. Compare the user's solution to an expert solution and provide feedback."
            ),
            HumanMessagePromptTemplate.from_template(
                "Scenario: {scenario}\n\n"
                "User's solution:\n{user_solution}\n\n"
                "Expert solution:\n{expert_solution}\n\n"
                "Provide feedback on the user's solution. Highlight what they got correct and what needs improvement.\n"
                "Rate their understanding on a scale of 1-10."
            )
        ])
        
        # Initialize feedback chain in Arabic
        self.feedback_template_ar = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "أنت خبير في معايير التمويل الإسلامي. قارن حل المستخدم بحل الخبير وقدم تعليقات."
            ),
            HumanMessagePromptTemplate.from_template(
                "السيناريو: {scenario}\n\n"
                "حل المستخدم:\n{user_solution}\n\n"
                "حل الخبير:\n{expert_solution}\n\n"
                "قدم تعليقات على حل المستخدم. سلط الضوء على ما أصابوه بشكل صحيح وما يحتاج إلى تحسين.\n"
                "قيّم فهمهم على مقياس من 1 إلى 10."
            )
        ])
        
        # Initialize chains
        self.explanation_chain_en = LLMChain(
            llm=self.chat_model,
            prompt=self.explanation_template_en
        )
        
        self.explanation_chain_ar = LLMChain(
            llm=self.chat_model,
            prompt=self.explanation_template_ar
        )
        
        self.feedback_chain = LLMChain(
            llm=self.chat_model,
            prompt=self.feedback_template
        )
        
        self.feedback_chain_ar = LLMChain(
            llm=self.chat_model,
            prompt=self.feedback_template_ar
        )
    
    def get_explanation(self, standard, standard_title, scenario, language="English"):
        """Get AI explanation for a specific standard and scenario"""
        if language == "English":
            return self.explanation_chain_en.run(
                standard_title=standard_title,
                scenario=scenario
            )
        else:  # Arabic
            return self.explanation_chain_ar.run(
                standard_title=standard_title,
                scenario=scenario
            )
    
    def get_feedback(self, scenario, user_solution, expert_solution, language="English"):
        """Get feedback on user's solution"""
        if language == "English":
            return self.feedback_chain.run(
                scenario=scenario,
                user_solution=user_solution,
                expert_solution=expert_solution
            )
        else:  # Arabic
            return self.feedback_chain_ar.run(
                scenario=scenario,
                user_solution=user_solution,
                expert_solution=expert_solution
            )

def generate_glossary(language):
    """Generate a glossary of Islamic finance terms"""
    
    terms = {
        "Ijarah": {
            "en": "A lease contract where one party transfers the right to use an asset to another party for an agreed period at an agreed consideration.",
            "ar": "عقد إيجار حيث ينقل طرف حق استخدام أصل إلى طرف آخر لفترة متفق عليها بمقابل متفق عليه."
        },
        "Murabaha": {
            "en": "A sales contract where the seller expressly mentions the cost incurred on the sold commodity and sells it to another person by adding some profit.",
            "ar": "عقد بيع حيث يذكر البائع صراحةً التكلفة التي تكبدها على السلعة المباعة ويبيعها لشخص آخر بإضافة بعض الربح."
        },
        "Wakala": {
            "en": "An agency contract where one party appoints another party to act on their behalf for a specific task.",
            "ar": "عقد وكالة حيث يعين طرف طرفًا آخر للتصرف نيابة عنه لمهمة محددة."
        },
        "Istisna'a": {
            "en": "A contract of sale where a commodity is transacted before it comes into existence, requiring the manufacturer to make it with payment from the buyer either in advance or by installments.",
            "ar": "عقد بيع حيث يتم تداول سلعة قبل وجودها، مما يتطلب من المصنع صنعها مع دفع المشتري إما مقدمًا أو على أقساط."
        },
        "Sukuk": {
            "en": "Islamic financial certificates, similar to bonds, that comply with Shariah law.",
            "ar": "شهادات مالية إسلامية، مشابهة للسندات، تتوافق مع الشريعة الإسلامية."
        }
    }
    
    if language == "English":
        return {term: terms[term]["en"] for term in terms}
    else:  # Arabic
        return {term: terms[term]["ar"] for term in terms}

def main():
    st.set_page_config(page_title="Islamic Finance Standards Simplified", layout="wide")
    
    # Initialize explanations class
    explanations = IslamicFinanceStandardsExplainer()
    
    # Initialize session state for memory
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(return_messages=True)
    
    # Language selection
    language = st.sidebar.selectbox("Language / اللغة", ["English", "Arabic / العربية"])
    
    # Display title based on selected language
    if language == "English":
        st.title("Islamic Finance Standards Simplified")
        st.markdown("### Learn AAOIFI standards through simple examples")
    else:  # Arabic
        st.title("تبسيط معايير التمويل الإسلامي")
        st.markdown("### تعلم معايير هيئة المحاسبة والمراجعة للمؤسسات المالية الإسلامية من خلال أمثلة بسيطة")
    
    # Sidebar navigation
    if language == "English":
        st.sidebar.header("Navigation")
        page = st.sidebar.radio("Go to", ["Home", "Standards Explorer", "Interactive Tutorial", "Glossary", "Custom Question"])
    else:  # Arabic
        st.sidebar.header("التنقل")
        page = st.sidebar.radio("اذهب إلى", ["الصفحة الرئيسية", "مستكشف المعايير", "الدروس التفاعلية", "المصطلحات", "سؤال مخصص"])
    
    # Map Arabic page selections to English for processing
    if language == "Arabic / العربية":
        page_map = {
            "الصفحة الرئيسية": "Home",
            "مستكشف المعايير": "Standards Explorer",
            "الدروس التفاعلية": "Interactive Tutorial",
            "المصطلحات": "Glossary",
            "سؤال مخصص": "Custom Question"
        }
        page = page_map.get(page, page)
    
    # Page content
    if page == "Home":
        if language == "English":
            st.write("""
            ## Welcome to Islamic Finance Standards Simplified
            
            This application helps you understand AAOIFI Financial Accounting Standards 
            through simple explanations and practical examples. 
            
            ### Features:
            - Explore the five key standards: FAS 4, FAS 7, FAS 10, FAS 28, and FAS 32
            - Learn through real-world examples and cases
            - Interactive tutorials to test your understanding
            - Multilingual support (English and Arabic)
            - Ask custom questions about Islamic finance standards
            
            Start by selecting a section from the sidebar!
            """)
        else:  # Arabic
            st.write("""
            ## مرحبًا بكم في تبسيط معايير التمويل الإسلامي
            
            يساعدك هذا التطبيق على فهم معايير المحاسبة المالية لهيئة المحاسبة والمراجعة للمؤسسات المالية الإسلامية
            من خلال شروحات بسيطة وأمثلة عملية.
            
            ### الميزات:
            - استكشاف المعايير الخمسة الرئيسية: FAS 4، FAS 7، FAS 10، FAS 28، و FAS 32
            - التعلم من خلال أمثلة وحالات من العالم الحقيقي
            - دروس تفاعلية لاختبار فهمك
            - دعم متعدد اللغات (الإنجليزية والعربية)
            - اطرح أسئلة مخصصة حول معايير التمويل الإسلامي
            
            ابدأ باختيار قسم من الشريط الجانبي!
            """)
    
    elif page == "Standards Explorer":
        if language == "English":
            st.markdown("## AAOIFI Standards Explorer")
            st.markdown("Select a standard to learn more about it")
        else:  # Arabic
            st.markdown("## مستكشف معايير هيئة المحاسبة والمراجعة للمؤسسات المالية الإسلامية")
            st.markdown("حدد معيارًا لمعرفة المزيد عنه")
        
        # Standard selection
        lang_code = "en" if language == "English" else "ar"
        standard_titles = {std: standards[std][f"title_{lang_code}"] for std in standards}
        selected_standard = st.selectbox(
            "Standard" if language == "English" else "المعيار",
            options=list(standards.keys()),
            format_func=lambda x: f"{x} - {standard_titles[x]}"
        )
        
        # Display standard details
        st.markdown(f"### {selected_standard} - {standards[selected_standard][f'title_{lang_code}']}")
        st.markdown(standards[selected_standard][f'description_{lang_code}'])
        
        # Display example scenario
        st.markdown("#### " + examples[selected_standard][f'title_{lang_code}'])
        st.markdown(examples[selected_standard][f'scenario_{lang_code}'])
        
        # Get AI explanation
        if st.button("Get Explanation" if language == "English" else "الحصول على شرح"):
            with st.spinner("Generating explanation..." if language == "English" else "جاري إنشاء الشرح..."):
                explanation = explanations.get_explanation(
                    standard=selected_standard,
                    standard_title=standards[selected_standard][f'title_{lang_code}'],
                    scenario=examples[selected_standard][f'scenario_{lang_code}'],
                    language=language
                )
                st.markdown("### " + ("Explanation" if language == "English" else "الشرح"))
                st.markdown(explanation)
                
                # Save to session memory
                st.session_state.memory.save_context(
                    {"input": f"Explain {selected_standard}"}, 
                    {"output": explanation}
                )
    
    elif page == "Interactive Tutorial":
        if language == "English":
            st.markdown("## Interactive Tutorial")
            st.markdown("Test your understanding of Islamic finance standards")
        else:  # Arabic
            st.markdown("## الدروس التفاعلية")
            st.markdown("اختبر فهمك لمعايير التمويل الإسلامي")
        
        # Select a standard for the tutorial
        lang_code = "en" if language == "English" else "ar"
        standard_titles = {std: standards[std][f"title_{lang_code}"] for std in standards}
        tutorial_standard = st.selectbox(
            "Select a standard for tutorial" if language == "English" else "اختر معيارًا للدرس",
            options=list(standards.keys()),
            format_func=lambda x: f"{x} - {standard_titles[x]}"
        )
        
        # Show scenario
        st.markdown("### " + examples[tutorial_standard][f'title_{lang_code}'])
        st.markdown(examples[tutorial_standard][f'scenario_{lang_code}'])
        
        # User attempts solution
        user_solution = st.text_area(
            "Enter your solution" if language == "English" else "أدخل حلك",
            height=150
        )
        
        # Check solution
        if st.button("Check My Answer" if language == "English" else "تحقق من إجابتي"):
            with st.spinner("Analyzing your answer..." if language == "English" else "تحليل إجابتك..."):
                # Get expert solution
                expert_solution = explanations.get_explanation(
                    standard=tutorial_standard,
                    standard_title=standards[tutorial_standard][f'title_{lang_code}'],
                    scenario=examples[tutorial_standard][f'scenario_{lang_code}'],
                    language=language
                )
                
                # Compare with user solution
                try:
                    feedback = explanations.get_feedback(
                        scenario=examples[tutorial_standard][f'scenario_{lang_code}'],
                        user_solution=user_solution,
                        expert_solution=expert_solution,
                        language=language
                    )
                    
                    st.markdown("### " + ("Feedback" if language == "English" else "التعليق"))
                    st.markdown(feedback)
                    
                    # Show the expert solution
                    with st.expander("Expert Solution" if language == "English" else "حل الخبير"):
                        st.markdown(expert_solution)
                
                except Exception as e:
                    st.error(f"Error generating feedback: {str(e)}")
    
    elif page == "Glossary":
        if language == "English":
            st.markdown("## Islamic Finance Glossary")
            st.markdown("Key terms and concepts in Islamic finance")
        else:  # Arabic
            st.markdown("## مصطلحات التمويل الإسلامي")
            st.markdown("المصطلحات والمفاهيم الرئيسية في التمويل الإسلامي")
        
        # Generate glossary
        glossary = generate_glossary(language)
        
        # Display glossary
        for term, definition in glossary.items():
            st.markdown(f"**{term}**: {definition}")
            st.markdown("---")
    
    elif page == "Custom Question":
        if language == "English":
            st.markdown("## Ask Your Own Question")
            st.markdown("Ask any question about Islamic finance standards")
        else:  # Arabic
            st.markdown("## اطرح سؤالك الخاص")
            st.markdown("اطرح أي سؤال حول معايير التمويل الإسلامي")
        
        # Custom question interface
        custom_question = st.text_area(
            "Your question" if language == "English" else "سؤالك",
            height=100
        )
        
        if st.button("Get Answer" if language == "English" else "الحصول على إجابة"):
            with st.spinner("Generating answer..." if language == "English" else "جاري إنشاء الإجابة..."):
                # Create custom question template
                if language == "English":
                    template = """
                    You are an expert in Islamic Finance standards, particularly AAOIFI standards.
                    The user has asked the following question about Islamic Finance:
                    
                    {question}
                    
                    Provide a clear, detailed answer using your knowledge of Islamic finance principles and standards.
                    Reference specific AAOIFI standards when relevant. Make your explanation easy for non-specialists to understand.
                    """
                else:  # Arabic
                    template = """
                    أنت خبير في معايير التمويل الإسلامي، خاصة معايير هيئة المحاسبة والمراجعة للمؤسسات المالية الإسلامية.
                    طرح المستخدم السؤال التالي حول التمويل الإسلامي:
                    
                    {question}
                    
                    قدم إجابة واضحة ومفصلة باستخدام معرفتك بمبادئ ومعايير التمويل الإسلامي.
                    أشر إلى معايير هيئة المحاسبة والمراجعة للمؤسسات المالية الإسلامية المحددة عندما يكون ذلك ذا صلة. اجعل شرحك سهلاً لغير المتخصصين لفهمه.
                    """
                
                prompt_template = PromptTemplate(
                    input_variables=["question"],
                    template=template
                )
                
                # Create chain for custom questions
                custom_chain = LLMChain(
                    llm=explanations.chat_model,
                    prompt=prompt_template
                )
                
                # Get answer
                answer = custom_chain.run(question=custom_question)
                
                st.markdown("### " + ("Answer" if language == "English" else "الإجابة"))
                st.markdown(answer)
                
                # Save to session memory
                st.session_state.memory.save_context(
                    {"input": custom_question}, 
                    {"output": answer}
                )

if __name__ == "__main__":
    main()
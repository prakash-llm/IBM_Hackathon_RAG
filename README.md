# CalPERS Multi-Agent Assistant

This project implements a multi-agent system using IBM watsonx services and CrewAI to provide intelligent responses to CalPERS members and employers' queries.

## Features

- Multi-agent orchestration using CrewAI
- IBM watsonx integration for embeddings and retrieval
- PDF-based knowledge base for members and employers FAQs
- Google search integration for general queries
- Human feedback loop
- Caching for frequently asked questions
- Comprehensive logging and error handling
- Streamlit-based user interface

## Project Structure

```
├── src/
│   ├── agents/           # Agent definitions and configurations
│   ├── knowledge_base/   # PDF processing and knowledge base management
│   ├── utils/           # Utility functions and helpers
│   └── app.py           # Main Streamlit application
├── config/              # Configuration files
├── logs/               # Application logs
├── requirements.txt    # Project dependencies
└── README.md          # Project documentation
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
Create a `.env` file with the following variables:
```
IBM_API_KEY=your_api_key
IBM_URL=your_watsonx_url
```

3. Run the application:
```bash
streamlit run src/app.py
```

## Usage

1. Access the application through your web browser
2. Select whether you're a member or employer
3. Ask your question in the chat interface
4. Provide feedback on responses to improve the system

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
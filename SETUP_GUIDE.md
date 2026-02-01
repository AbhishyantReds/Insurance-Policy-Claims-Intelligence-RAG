# Quick Setup & Deployment Guide

## Quick Start (Local)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your OpenAI API key
export OPENAI_API_KEY="your-key-here"  # Linux/Mac
# or
set OPENAI_API_KEY=your-key-here       # Windows

# 3. Place policy PDFs in data/insurance_policies/

# 4. Run the app
python app.py
# Visit http://localhost:7860

# OR run as API + Frontend
python -m uvicorn app.main:app --reload  # Terminal 1 - Backend :8000
python gradio_app.py                      # Terminal 2 - Frontend :7860
```

## Docker Deployment

```bash
# 1. Set API key in .env file
echo "OPENAI_API_KEY=your-key-here" > .env

# 2. Build and run
docker-compose up --build

# 3. Access services
# - Backend API: http://localhost:8000
# - Frontend UI: http://localhost:7860
# - API Docs: http://localhost:8000/docs
# - Metrics: http://localhost:8000/metrics

# 4. Stop services
docker-compose down
```

## First Time Usage

1. **Upload Documents**: Place insurance policy PDFs in `data/insurance_policies/`

2. **Ingest Documents**: 
   - Open Gradio UI (http://localhost:7860)
   - Go to "Admin" tab
   - Click "Ingest Documents"
   - Wait for completion (creates vector DB + BM25 index)

3. **Query Policies**:
   - Go to "Ask Questions" tab
   - Enter: "What is my homeowner's insurance deductible?"
   - Review answer with confidence score

4. **Check Metrics**:
   ```bash
   curl http://localhost:8000/metrics?days=7
   ```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific tests
pytest tests/test_validation.py -v
pytest tests/test_retrieval.py -v

# With coverage
pytest --cov=app tests/
```

## Troubleshooting

### Issue: "No relevant documents found"
**Solution**: 
1. Check if documents are in `data/insurance_policies/`
2. Re-run ingestion in Admin tab
3. Verify vectordb/ and BM25 index created

### Issue: "OpenAI API key not configured"
**Solution**: 
```bash
export OPENAI_API_KEY="sk-..."
# or add to .env file
```

### Issue: "BM25 index not found"
**Solution**: 
- Run document ingestion to create BM25 index
- Check that `vectordb/bm25_index.pkl` exists

### Issue: Docker containers won't start
**Solution**:
```bash
# Check logs
docker-compose logs backend

# Rebuild images
docker-compose build --no-cache

# Check ports not in use
netstat -ano | findstr :8000
netstat -ano | findstr :7860
```

## Production Checklist

- [ ] Set strong API key
- [ ] Enable rate limiting (add middleware)
- [ ] Add authentication (JWT/OAuth)
- [ ] Configure CORS properly
- [ ] Set up Redis caching
- [ ] Configure cloud storage for vectordb
- [ ] Set up monitoring alerts
- [ ] Enable HTTPS/SSL
- [ ] Set up backup for monitoring.db
- [ ] Configure log rotation
- [ ] Add CDN for static assets

## Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional
HYBRID_SEARCH_ENABLED=true
BM25_WEIGHT=0.5
SEMANTIC_WEIGHT=0.5
MONITORING_ENABLED=true
CONFIDENCE_THRESHOLD=0.6
MIN_RELEVANCE_SCORE=0.5
```

## Performance Tuning

### Adjust Retrieval
```python
# In app/config.py
DEFAULT_K_RESULTS = 6  # Increase for more context
CHUNK_SIZE = 1500      # Larger chunks = more context
CHUNK_OVERLAP = 200    # More overlap = better continuity
```

### Adjust Search Weighting
```python
# In app/config.py
BM25_WEIGHT = 0.6      # More weight to keyword search
SEMANTIC_WEIGHT = 0.4  # Less weight to semantic
```

### Adjust Quality Thresholds
```python
# In app/config.py
MIN_RELEVANCE_SCORE = 0.4  # Lower = more permissive
CONFIDENCE_THRESHOLD = 0.5  # Lower = fewer low-confidence warnings
```

## Monitoring Dashboard

Coming soon: Gradio dashboard tab with:
- Query volume over time
- Average confidence scores
- Response time histogram
- Most common questions
- Low confidence alerts
- Cost tracking

## API Examples

### Query with Confidence
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is my deductible?"}'
```

### Check Coverage
```bash
curl -X POST http://localhost:8000/check-coverage \
  -H "Content-Type: application/json" \
  -d '{"scenario": "My basement flooded"}'
```

### Get Metrics
```bash
curl http://localhost:8000/metrics?days=30
```

## Next Steps

1. ✅ Add your policy documents
2. ✅ Run ingestion
3. ✅ Test queries
4. ✅ Review confidence scores
5. ✅ Check metrics
6. 🚀 Deploy to production
7. 📊 Monitor performance
8. 🔄 Iterate on prompts based on metrics

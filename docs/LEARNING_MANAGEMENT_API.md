# Learning Management API Design

## Overview
The Learning Management API provides endpoints for monitoring, controlling, and managing Allie's self-learning capabilities. It integrates with the existing FastAPI backend to provide a comprehensive learning management system.

## API Endpoints

### Learning Status and Monitoring

#### GET /api/learning/status
Get current learning system status
```json
{
  "learning_enabled": true,
  "current_episode": {
    "id": "learn_20241107_001",
    "status": "training",
    "progress": 0.65,
    "start_time": "2024-11-07T14:30:00Z",
    "estimated_completion": "2024-11-07T15:00:00Z"
  },
  "last_episode": {
    "id": "learn_20241107_000",
    "status": "completed",
    "metrics": {
      "learning_gains": {"coherence": 0.12, "relevance": 0.08},
      "safety_score": 0.95,
      "forgetting_detected": false
    },
    "completion_time": "2024-11-07T14:00:00Z"
  },
  "system_health": {
    "data_quality": 0.78,
    "model_stability": 0.92,
    "resource_usage": 0.45
  }
}
```

#### GET /api/learning/history
Get learning episode history
**Query Parameters:**
- `limit` (int): Number of episodes to return (default: 10)
- `offset` (int): Pagination offset (default: 0)
- `status` (str): Filter by status (training, completed, failed)

```json
{
  "episodes": [
    {
      "id": "learn_20241107_001",
      "status": "completed",
      "start_time": "2024-11-07T13:00:00Z",
      "end_time": "2024-11-07T13:45:00Z",
      "metrics": {
        "perplexity": 42.3,
        "coherence": 0.76,
        "learning_gains": {"coherence": 0.05, "relevance": 0.03},
        "safety_score": 0.98
      },
      "training_data": {
        "sample_count": 150,
        "quality_score": 0.82,
        "topics": ["science", "history", "technology"]
      }
    }
  ],
  "total_count": 25,
  "summary": {
    "total_episodes": 25,
    "successful_episodes": 22,
    "average_learning_gain": 0.07,
    "forgetting_incidents": 1
  }
}
```

### Learning Control

#### POST /api/learning/trigger
Manually trigger a learning episode
**Request Body:**
```json
{
  "reason": "manual_trigger",
  "priority": "medium",
  "data_filters": {
    "min_quality": 0.7,
    "topics": ["science", "technology"],
    "max_samples": 200
  }
}
```
**Response:**
```json
{
  "episode_id": "learn_20241107_002",
  "status": "scheduled",
  "estimated_start": "2024-11-07T15:30:00Z",
  "data_available": 180
}
```

#### POST /api/learning/pause
Pause the current learning episode
**Response:**
```json
{
  "episode_id": "learn_20241107_001",
  "status": "paused",
  "can_resume": true
}
```

#### POST /api/learning/resume
Resume a paused learning episode
**Response:**
```json
{
  "episode_id": "learn_20241107_001",
  "status": "training",
  "progress": 0.65
}
```

#### POST /api/learning/cancel
Cancel the current learning episode
**Response:**
```json
{
  "episode_id": "learn_20241107_001",
  "status": "cancelled",
  "reason": "user_cancelled"
}
```

### Configuration Management

#### GET /api/learning/config
Get current learning configuration
```json
{
  "learning_enabled": true,
  "scheduling": {
    "data_threshold": 50,
    "quality_threshold": 0.8,
    "time_window_start": "02:00",
    "time_window_end": "06:00",
    "max_daily_episodes": 2
  },
  "training": {
    "max_epochs": 3,
    "batch_size": 4,
    "learning_rate": 0.0002,
    "replay_ratio": 0.3,
    "ewc_lambda": 0.1
  },
  "safety": {
    "min_safety_score": 0.8,
    "max_bias_score": 0.3,
    "content_filters": ["harmful", "pii", "bias"]
  }
}
```

#### PUT /api/learning/config
Update learning configuration
**Request Body:** (partial updates supported)
```json
{
  "scheduling": {
    "data_threshold": 75
  },
  "training": {
    "learning_rate": 0.00015
  }
}
```

### Data Management

#### GET /api/learning/data/stats
Get training data statistics
```json
{
  "total_conversations": 1250,
  "quality_distribution": {
    "excellent": 320,
    "good": 480,
    "fair": 350,
    "poor": 100
  },
  "topic_distribution": {
    "science": 280,
    "technology": 220,
    "history": 180,
    "general": 570
  },
  "temporal_distribution": {
    "last_24h": 45,
    "last_7d": 180,
    "last_30d": 650
  },
  "safety_stats": {
    "approved_rate": 0.87,
    "common_rejections": {
      "harmful_content": 45,
      "low_quality": 78,
      "bias_detected": 12
    }
  }
}
```

#### POST /api/learning/data/validate
Validate a conversation for training
**Request Body:**
```json
{
  "prompt": "What is machine learning?",
  "response": "Machine learning is a subset of artificial intelligence..."
}
```
**Response:**
```json
{
  "approved": true,
  "scores": {
    "safety": 0.98,
    "quality": {
      "coherence": 0.85,
      "relevance": 0.92,
      "informativeness": 0.78
    },
    "bias": 0.05
  },
  "reason": "Approved for training"
}
```

#### DELETE /api/learning/data/{conversation_id}
Remove a conversation from training data
**Response:**
```json
{
  "conversation_id": "conv_12345",
  "status": "removed",
  "affected_episodes": ["learn_20241107_001"]
}
```

### Model Management

#### GET /api/learning/models
List available model versions
```json
{
  "current_model": "v1.2",
  "models": [
    {
      "version": "v1.2",
      "created": "2024-11-07T14:00:00Z",
      "metrics": {
        "perplexity": 42.3,
        "coherence": 0.76,
        "safety_score": 0.95
      },
      "training_episodes": 15,
      "status": "active"
    },
    {
      "version": "v1.1",
      "created": "2024-11-06T10:00:00Z",
      "metrics": {
        "perplexity": 45.1,
        "coherence": 0.72,
        "safety_score": 0.97
      },
      "training_episodes": 12,
      "status": "backup"
    }
  ]
}
```

#### POST /api/learning/models/rollback
Rollback to a previous model version
**Request Body:**
```json
{
  "target_version": "v1.1",
  "reason": "Performance degradation detected"
}
```
**Response:**
```json
{
  "status": "rollback_initiated",
  "target_version": "v1.1",
  "estimated_completion": "2024-11-07T15:05:00Z"
}
```

### Analytics and Insights

#### GET /api/learning/analytics/learning-progress
Get learning progress analytics
```json
{
  "time_range": "30d",
  "metrics": {
    "response_quality_trend": [
      {"date": "2024-10-08", "coherence": 0.65, "relevance": 0.58},
      {"date": "2024-10-15", "coherence": 0.68, "relevance": 0.62},
      {"date": "2024-10-22", "coherence": 0.71, "relevance": 0.65}
    ],
    "learning_efficiency": 0.0042,
    "knowledge_domains": {
      "improving": ["science", "technology"],
      "stable": ["history", "literature"],
      "needs_attention": ["current_events"]
    }
  },
  "insights": [
    "Response coherence improved by 9% over the last month",
    "Technology domain shows strongest learning gains",
    "Current events knowledge needs more training data"
  ]
}
```

#### GET /api/learning/analytics/user-satisfaction
Get user satisfaction metrics
```json
{
  "overall_satisfaction": 0.82,
  "response_ratings": {
    "excellent": 45,
    "good": 120,
    "fair": 35,
    "poor": 8
  },
  "common_feedback": [
    "More detailed explanations",
    "Better handling of complex questions",
    "Faster response times"
  ],
  "satisfaction_trends": [
    {"week": "2024-W44", "satisfaction": 0.78},
    {"week": "2024-W45", "satisfaction": 0.81},
    {"week": "2024-W46", "satisfaction": 0.82}
  ]
}
```

## Implementation Considerations

### Authentication and Authorization
- **API Key Authentication**: Required for all learning management endpoints
- **Role-Based Access**: Admin access for configuration, read-only for monitoring
- **Rate Limiting**: Prevent abuse of manual trigger endpoints

### Error Handling
- **Structured Error Responses**: Consistent error format across all endpoints
- **Graceful Degradation**: API remains functional even if learning system is down
- **Detailed Logging**: Comprehensive logging for debugging and auditing

### Performance Optimization
- **Caching**: Cache expensive computations (analytics, model metrics)
- **Async Processing**: Non-blocking operations for long-running tasks
- **Pagination**: Efficient handling of large datasets
- **Compression**: Gzip compression for large response payloads

### Monitoring and Alerting
- **Health Checks**: Automatic monitoring of API endpoints
- **Performance Metrics**: Response times, error rates, throughput
- **Business Metrics**: Learning episode success rates, user satisfaction trends

## Integration with Frontend

### Status Dashboard
- Real-time learning status updates
- Progress bars for active learning episodes
- Historical performance charts

### Configuration Panel
- User-friendly controls for learning settings
- Visual feedback for configuration changes
- Help tooltips and documentation links

### Analytics Dashboard
- Interactive charts showing learning progress
- Drill-down capabilities for detailed metrics
- Export functionality for reports

## Security Considerations

### Data Protection
- **Encryption**: All data encrypted at rest and in transit
- **Access Controls**: Strict permissions for sensitive operations
- **Audit Logging**: Complete audit trail of all API operations

### Model Security
- **Validation**: All model updates validated before deployment
- **Rollback Capability**: Quick reversion to safe model versions
- **Integrity Checks**: Hash verification for model files

### User Privacy
- **Anonymization**: Training data anonymized before processing
- **Consent Management**: Clear user consent for learning participation
- **Data Deletion**: Complete removal of user data when requested

## Future Extensions

### Advanced Analytics
- **Predictive Modeling**: Forecast learning effectiveness
- **A/B Testing**: Compare different learning strategies
- **Personalization**: User-specific learning preferences

### Integration APIs
- **Webhook Support**: Notify external systems of learning events
- **Plugin Architecture**: Extensible learning strategies
- **Multi-Model Support**: Manage multiple specialized models

This API design provides comprehensive control and monitoring of Allie's learning system while maintaining safety, performance, and usability.
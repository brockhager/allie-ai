# Fact-Check UI

A React-based interface for human verification and management of Allie's learning pipeline.

## Features

- **Fact Table**: View all facts with status, confidence scores, and metadata
- **Filtering**: Filter facts by status, category, and minimum confidence
- **Fact Details Panel**: Detailed view of selected facts with provenance and history
- **Bulk Actions**: Approve, reject, or mark facts for review
- **Feature Flags**: Visual indicators for system status (read-only mode, auto-apply)
- **Tabbed Navigation**: Switch between Allie UI and Fact Checking interface

## Setup

1. Install dependencies:
```bash
npm install
```

2. Start development server:
```bash
npm start
```

3. Build for production:
```bash
npm run build
```

## API Integration

The UI expects the following backend endpoints:

- `GET /api/facts` - List facts with optional filters
- `PATCH /api/facts/:id` - Update fact status and confidence
- `POST /api/facts/bulk-update` - Bulk update multiple facts
- `GET /api/feature_flags` - Get current feature flag values

## Usage

1. **Navigation**: Use tabs to switch between Allie UI and Fact Checking
2. **Filtering**: Use filters to narrow down facts by status or confidence
3. **Review**: Click on a fact row to view details in the side panel
4. **Actions**: Use action buttons to approve/reject facts or mark for review
5. **Bulk Operations**: Select multiple facts for bulk status updates

## Status Types

- **true**: Verified correct fact
- **false**: Verified incorrect fact
- **not_verified**: Not yet reviewed
- **needs_review**: Requires additional investigation
- **experimental**: New fact being tested

## Confidence Scoring

- **80-100%**: High confidence (green)
- **60-79%**: Medium confidence (yellow)
- **40-59%**: Low confidence (orange)
- **0-39%**: Very low confidence (red)

## Safety Features

- Read-only mode prevents accidental changes
- All actions are logged for audit trails
- Human verification required for destructive operations
- Feature flags control system behavior
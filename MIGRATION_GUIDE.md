# Migration Guide: Self-Hosted Infrastructure

This guide explains how to migrate from Supabase + Vercel Blob to self-hosted PostgreSQL + MinIO.

## Overview

**Before:** Supabase (PostgreSQL) + Vercel Blob (File Storage)  
**After:** Self-hosted PostgreSQL 17 + MinIO (S3-compatible storage)

## Prerequisites

- Server with PostgreSQL 17 installed
- MinIO running (Docker or standalone)
- Node.js 18+ for running migration scripts

## Step 1: Environment Variables

Update your `.env.local` file with new variables:

\`\`\`bash
# PostgreSQL
DATABASE_URL=postgresql://galeries_user:PASSWORD@localhost:5432/galleries

# MinIO
S3_ENDPOINT=http://localhost:9200
S3_ACCESS_KEY=galleries-app
S3_SECRET_KEY=your_secret_key
S3_BUCKET=galleries
S3_REGION=us-east-1
S3_PUBLIC_URL=https://s3.vlcpadel.com
\`\`\`

## Step 2: Data Layer Changes

The application now uses:

1. **PostgreSQL Client** (`lib/database/server.ts`)
   - Replaces `@supabase/ssr` with `postgres` package
   - Direct SQL queries instead of Supabase query builder
   
2. **MinIO/S3 Client** (`lib/storage/s3client.ts`)
   - Replaces `@vercel/blob` with `@aws-sdk/client-s3`
   - S3-compatible API for file operations

## Step 3: Backward Compatibility

During migration, both old and new systems work simultaneously:

- `lib/supabase/server.ts` - Still works with Supabase (legacy)
- `lib/database/server.ts` - New PostgreSQL client (recommended)

**Migration strategy:**
1. Keep Supabase env vars during testing
2. Gradually replace Supabase queries with direct PostgreSQL
3. Remove Supabase env vars after full migration

## Step 4: Testing

Test the new data layer:

\`\`\`bash
# Test PostgreSQL connection
PGPASSWORD='your_password' psql -h localhost -U galeries_user -d galleries -c "SELECT COUNT(*) FROM galleries;"

# Test MinIO upload
curl -X POST http://localhost:3000/api/upload \
  -F "file=@test-image.jpg"
\`\`\`

## Step 5: Deployment

1. Update production environment variables
2. Deploy application with new code
3. Monitor logs for any connection issues
4. Remove legacy Supabase/Blob code after confirming stability

## Rollback Plan

If issues occur:
1. Restore old environment variables (Supabase + Blob)
2. The app falls back to legacy clients automatically
3. Fix issues in staging before retry

## Files Changed

**New files:**
- `lib/database/server.ts` - PostgreSQL client
- `lib/storage/s3client.ts` - MinIO/S3 client

**Modified files:**
- `app/api/upload/route.ts` - Uses new S3 client
- `lib/supabase/server.ts` - Added `getDbClient()` helper
- `.env.example` - Added new variables

**No changes to:**
- UI components
- Business logic
- API routes structure
- Authentication flow

## Notes

- Legacy Face-API fields (bounding_box, descriptor, confidence) were NOT migrated
- Only InsightFace data (512-dimension descriptors) was transferred
- All file URLs in database were updated to MinIO URLs during migration

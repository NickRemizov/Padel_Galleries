# Deployment Checklist for Self-Hosted Migration

## Pre-Deployment

- [ ] PostgreSQL 17 installed and running
- [ ] MinIO installed and accessible
- [ ] pgvector extension installed in PostgreSQL
- [ ] Database schema created (`001_create_schema.sql`)
- [ ] Data migrated from Supabase (`002_migrate_data.ts`)
- [ ] Files migrated from Vercel Blob (`003_migrate_files.ts`)

## Environment Variables (Production)

Add these to Vercel project settings or server `.env`:

\`\`\`bash
DATABASE_URL=postgresql://galeries_user:PASSWORD@YOUR_SERVER_IP:5432/galleries
S3_ENDPOINT=http://YOUR_SERVER_IP:9200
S3_ACCESS_KEY=galleries-app
S3_SECRET_KEY=YOUR_SECRET
S3_BUCKET=galleries
S3_REGION=us-east-1
S3_PUBLIC_URL=https://s3.vlcpadel.com
FASTAPI_URL=https://api.vlcpadel.com
NEXT_PUBLIC_FASTAPI_URL=https://api.vlcpadel.com
\`\`\`

## Deployment Steps

1. **Install dependencies:**
   \`\`\`bash
   npm install
   \`\`\`

2. **Build application:**
   \`\`\`bash
   npm run build
   \`\`\`

3. **Test locally with new env vars:**
   \`\`\`bash
   npm run dev
   \`\`\`

4. **Deploy to server:**
   \`\`\`bash
   # Option 1: Vercel (update env vars first)
   vercel --prod
   
   # Option 2: Self-hosted
   pm2 start npm --name "galleries" -- start
   \`\`\`

## Post-Deployment Verification

### 1. Test Database Connection
\`\`\`bash
# SSH into server
PGPASSWORD='password' psql -h localhost -U galeries_user -d galleries

# Run test query
SELECT COUNT(*) FROM galleries;
SELECT COUNT(*) FROM gallery_images;
SELECT COUNT(*) FROM people;
\`\`\`

### 2. Test File Upload
1. Go to admin panel: `https://vlcpadel.com/admin`
2. Create test gallery
3. Upload test image
4. Verify image appears and loads correctly

### 3. Test Face Recognition
1. Upload gallery with faces
2. Check FastAPI processes images: `https://api.vlcpadel.com/health`
3. Verify faces detected in admin panel

### 4. Check Logs
\`\`\`bash
# Application logs
pm2 logs galleries

# PostgreSQL logs
sudo tail -f /var/log/postgresql/postgresql-17-main.log

# MinIO logs
docker logs minio -f
\`\`\`

## Troubleshooting

### Database connection fails
- Check `DATABASE_URL` format
- Verify PostgreSQL allows remote connections (`pg_hba.conf`)
- Test connection: `psql $DATABASE_URL`

### File upload fails
- Check MinIO is running: `curl http://localhost:9200/minio/health/live`
- Verify S3 credentials are correct
- Check bucket exists: `mc ls local/galleries`

### Images don't load
- Check `S3_PUBLIC_URL` points to correct domain
- Verify Traefik routes MinIO correctly
- Test direct URL: `curl https://s3.vlcpadel.com/galleries/test.jpg`

## Rollback Procedure

If critical issues occur:

1. **Restore old env vars** (Supabase + Vercel Blob)
2. **Redeploy previous version:**
   \`\`\`bash
   git revert HEAD
   vercel --prod
   \`\`\`
3. **Verify service restored**
4. **Debug issues in staging**

## Success Criteria

- [ ] Admin login works
- [ ] Galleries load correctly
- [ ] Images display properly
- [ ] File upload succeeds
- [ ] Face recognition processes images
- [ ] No errors in logs for 24 hours
- [ ] Performance is acceptable (< 2s page load)

## Next Steps After Stable Migration

1. Remove Supabase env vars
2. Delete Supabase integration
3. Cancel Vercel Blob subscription
4. Update documentation
5. Setup automated backups for PostgreSQL and MinIO

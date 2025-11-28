/**
 * File Migration Script: Vercel Blob → MinIO
 * 
 * This script migrates all images from Vercel Blob to your MinIO server
 * 
 * Usage:
 *   tsx migration-scripts/003_migrate_files.ts
 */

import { S3Client, PutObjectCommand, HeadBucketCommand } from '@aws-sdk/client-s3'
import postgres from 'postgres'

const s3Client = new S3Client({
  endpoint: process.env.S3_ENDPOINT || 'http://localhost:9000',
  region: process.env.S3_REGION || 'us-east-1',
  credentials: {
    accessKeyId: process.env.S3_ACCESS_KEY || '',
    secretAccessKey: process.env.S3_SECRET_KEY || '',
  },
  forcePathStyle: true, // Required for MinIO
})

const BUCKET_NAME = process.env.S3_BUCKET || 'galleries'
const BLOB_TOKEN = process.env.BLOB_READ_WRITE_TOKEN || ''

// Database connection to update URLs
const db = postgres(process.env.TARGET_DATABASE_URL!, {
  max: 10,
})

interface FileToMigrate {
  id: string
  image_url: string
  original_url: string | null
}

interface MigrationResult {
  total: number
  success: number
  failed: number
  skipped: number
}

async function checkBucket() {
  try {
    await s3Client.send(new HeadBucketCommand({ Bucket: BUCKET_NAME }))
    console.log(`✅ Bucket "${BUCKET_NAME}" exists`)
  } catch (error) {
    console.error(`❌ Bucket "${BUCKET_NAME}" not found!`)
    throw error
  }
}

async function downloadFromBlob(url: string): Promise<Buffer> {
  const response = await fetch(url, {
    headers: {
      Authorization: `Bearer ${BLOB_TOKEN}`,
    },
  })

  if (!response.ok) {
    throw new Error(`Failed to download from Blob: ${response.statusText}`)
  }

  const arrayBuffer = await response.arrayBuffer()
  return Buffer.from(arrayBuffer)
}

async function uploadToMinIO(buffer: Buffer, key: string, contentType: string): Promise<string> {
  await s3Client.send(
    new PutObjectCommand({
      Bucket: BUCKET_NAME,
      Key: key,
      Body: buffer,
      ContentType: contentType,
    })
  )

  // Return public URL
  const publicUrl = `${process.env.S3_PUBLIC_URL || 'https://s3.vlcpadel.com/galleries'}/${key}`
  return publicUrl
}

function extractFilenameFromUrl(url: string): string {
  const parts = url.split('/')
  return parts[parts.length - 1]
}

function getContentType(filename: string): string {
  const ext = filename.split('.').pop()?.toLowerCase()
  const contentTypes: Record<string, string> = {
    jpg: 'image/jpeg',
    jpeg: 'image/jpeg',
    png: 'image/png',
    gif: 'image/gif',
    webp: 'image/webp',
  }
  return contentTypes[ext || ''] || 'application/octet-stream'
}

async function migrateImages(tableName: string, urlField: string): Promise<MigrationResult> {
  console.log(`\n📦 Migrating images from ${tableName}.${urlField}...`)

  const result: MigrationResult = {
    total: 0,
    success: 0,
    failed: 0,
    skipped: 0,
  }

  try {
    // Get all rows with Blob URLs
    const rows = await db`
      SELECT id, ${db(urlField)} as url
      FROM ${db(tableName)}
      WHERE ${db(urlField)} LIKE 'https://%.blob.vercel-storage.com/%'
    `

    result.total = rows.length
    console.log(`  Found ${result.total} images to migrate`)

    if (result.total === 0) {
      console.log(`  ✅ No images to migrate`)
      return result
    }

    for (let i = 0; i < rows.length; i++) {
      const row = rows[i]
      process.stdout.write(`\r  Progress: ${i + 1}/${result.total}`)

      try {
        const blobUrl = row.url
        const filename = extractFilenameFromUrl(blobUrl)
        const contentType = getContentType(filename)

        // Download from Blob
        const buffer = await downloadFromBlob(blobUrl)

        // Generate MinIO key (year/month/filename structure)
        const date = new Date()
        const year = date.getFullYear()
        const month = String(date.getMonth() + 1).padStart(2, '0')
        const minioKey = `${year}/${month}/${filename}`

        // Upload to MinIO
        const newUrl = await uploadToMinIO(buffer, minioKey, contentType)

        // Update database
        await db`
          UPDATE ${db(tableName)}
          SET ${db(urlField)} = ${newUrl}
          WHERE id = ${row.id}
        `

        result.success++
      } catch (error) {
        console.error(`\n  ❌ Failed to migrate ${row.id}:`, error)
        result.failed++
      }
    }

    console.log(`\n  ✅ Migrated ${result.success}/${result.total} images`)
  } catch (error) {
    console.error(`  ❌ Error migrating ${tableName}:`, error)
  }

  return result
}

async function main() {
  console.log('🚀 Starting file migration from Vercel Blob to MinIO\n')
  console.log('Target:', process.env.S3_ENDPOINT)
  console.log('Bucket:', BUCKET_NAME)
  console.log('Public URL:', process.env.S3_PUBLIC_URL)

  try {
    // Test connections
    console.log('\n🔌 Testing connections...')
    await checkBucket()
    await db`SELECT 1`
    console.log('✅ Database connected')

    // Migrate images from different tables
    const results: MigrationResult[] = []

    // 1. Gallery images
    results.push(await migrateImages('gallery_images', 'image_url'))
    results.push(await migrateImages('gallery_images', 'original_url'))

    // 2. Gallery covers
    results.push(await migrateImages('galleries', 'cover_image_url'))
    results.push(await migrateImages('galleries', 'cover_image_square_url'))

    // 3. People avatars
    results.push(await migrateImages('people', 'avatar_url'))

    // 4. User photos
    results.push(await migrateImages('users', 'photo_url'))

    // Print summary
    console.log('\n\n📊 Migration Summary:')
    console.log('═'.repeat(60))
    
    const totalFiles = results.reduce((sum, r) => sum + r.total, 0)
    const totalSuccess = results.reduce((sum, r) => sum + r.success, 0)
    const totalFailed = results.reduce((sum, r) => sum + r.failed, 0)
    const totalSkipped = results.reduce((sum, r) => sum + r.skipped, 0)
    
    console.log(`Total files: ${totalFiles}`)
    console.log(`✅ Success: ${totalSuccess}`)
    console.log(`❌ Failed: ${totalFailed}`)
    console.log(`⏭️  Skipped: ${totalSkipped}`)
    console.log('═'.repeat(60))

    if (totalFailed > 0) {
      console.log('\n⚠️  Some files failed to migrate. Check logs above.')
    } else {
      console.log('\n✅ All files migrated successfully!')
    }
  } catch (error) {
    console.error('\n❌ Migration failed:', error)
    process.exit(1)
  } finally {
    await db.end()
  }
}

main()

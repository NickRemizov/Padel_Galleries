/**
 * Data Migration Script: Supabase → Self-hosted PostgreSQL
 * 
 * This script migrates all data from Supabase to your PostgreSQL server
 * Excludes legacy Face-API fields (bounding_box, descriptor, confidence)
 * 
 * Usage:
 *   tsx migration-scripts/002_migrate_data.ts
 */

import postgres from 'postgres'

// Source: Supabase
const sourceDb = postgres(process.env.SUPABASE_DB_URL!, {
  ssl: 'require',
  max: 10,
})

// Target: Your PostgreSQL
const targetDb = postgres(process.env.TARGET_DATABASE_URL!, {
  max: 10,
})

interface MigrationStats {
  table: string
  sourceCount: number
  targetCount: number
  skippedCount: number
  duration: number
}

const stats: MigrationStats[] = []

async function migrateTable(
  tableName: string,
  selectFields: string = '*',
  transformFn?: (row: any) => any
) {
  console.log(`\n📦 Migrating ${tableName}...`)
  const startTime = Date.now()

  try {
    // Get source data
    const sourceData = await sourceDb`
      SELECT ${sourceDb.unsafe(selectFields)}
      FROM ${sourceDb(tableName)}
    `
    
    console.log(`  Found ${sourceData.length} rows in source`)

    if (sourceData.length === 0) {
      stats.push({
        table: tableName,
        sourceCount: 0,
        targetCount: 0,
        skippedCount: 0,
        duration: Date.now() - startTime,
      })
      console.log(`  ✅ No data to migrate`)
      return
    }

    // Transform data if needed
    let targetData = transformFn ? sourceData.map(transformFn) : sourceData
    
    // Filter out rows with null/undefined values for required fields
    const originalCount = targetData.length
    targetData = targetData.filter(row => {
      // Basic validation - ensure id exists
      return row.id !== null && row.id !== undefined
    })
    const skippedCount = originalCount - targetData.length

    if (targetData.length === 0) {
      console.log(`  ⚠️  All rows filtered out, skipping`)
      stats.push({
        table: tableName,
        sourceCount: sourceData.length,
        targetCount: 0,
        skippedCount,
        duration: Date.now() - startTime,
      })
      return
    }

    // Insert in batches
    const batchSize = 100
    let inserted = 0

    for (let i = 0; i < targetData.length; i += batchSize) {
      const batch = targetData.slice(i, i + batchSize)
      
      await targetDb`
        INSERT INTO ${targetDb(tableName)}
        ${targetDb(batch)}
        ON CONFLICT (id) DO UPDATE SET
        ${targetDb(batch[0], ...Object.keys(batch[0]).filter(k => k !== 'id'))}
      `
      
      inserted += batch.length
      process.stdout.write(`\r  Inserted ${inserted}/${targetData.length} rows`)
    }

    console.log(`\n  ✅ Migrated ${inserted} rows (skipped ${skippedCount})`)
    
    stats.push({
      table: tableName,
      sourceCount: sourceData.length,
      targetCount: inserted,
      skippedCount,
      duration: Date.now() - startTime,
    })
  } catch (error) {
    console.error(`  ❌ Error migrating ${tableName}:`, error)
    stats.push({
      table: tableName,
      sourceCount: 0,
      targetCount: 0,
      skippedCount: 0,
      duration: Date.now() - startTime,
    })
  }
}

async function main() {
  console.log('🚀 Starting data migration from Supabase to Self-hosted PostgreSQL\n')
  console.log('Source:', process.env.SUPABASE_DB_URL?.substring(0, 50) + '...')
  console.log('Target:', process.env.TARGET_DATABASE_URL?.substring(0, 50) + '...')

  try {
    // Test connections
    console.log('\n🔌 Testing database connections...')
    await sourceDb`SELECT 1`
    console.log('  ✅ Source DB connected')
    await targetDb`SELECT 1`
    console.log('  ✅ Target DB connected')

    // Migrate in dependency order
    
    // 1. Independent tables (no foreign keys)
    await migrateTable('photographers')
    await migrateTable('locations')
    await migrateTable('organizers')
    await migrateTable('users')

    // 2. Galleries (depends on photographers, locations, organizers)
    await migrateTable('galleries')

    // 3. Gallery images (depends on galleries)
    await migrateTable('gallery_images')

    // 4. People
    await migrateTable('people')

    // 5. Photo faces (exclude legacy Face-API fields)
    await migrateTable(
      'photo_faces',
      `
        id, photo_id, person_id, face_category,
        insightface_descriptor, insightface_confidence, insightface_bbox, insightface_det_score,
        recognition_confidence, training_used, training_context,
        blur_score, verified, verified_at, verified_by,
        created_at, updated_at
      `.replace(/\s+/g, ' ').trim(),
      (row) => {
        // Convert vector to proper format if needed
        return row
      }
    )

    // 6. Face descriptors
    await migrateTable('face_descriptors')

    // 7. Face training sessions
    await migrateTable('face_training_sessions')

    // 8. Face recognition config
    await migrateTable('face_recognition_config')

    // 9. Rejected faces
    await migrateTable('rejected_faces')

    // 10. Gallery co-occurrence
    await migrateTable('gallery_co_occurrence')

    // 11. Tournament results
    await migrateTable('tournament_results')

    // 12. User interactions
    await migrateTable('likes')
    await migrateTable('favorites')
    await migrateTable('comments')

    // Print summary
    console.log('\n\n📊 Migration Summary:')
    console.log('═'.repeat(80))
    console.log(
      `${'Table'.padEnd(30)} ${'Source'.padStart(10)} ${'Target'.padStart(10)} ${'Skipped'.padStart(10)} ${'Time (ms)'.padStart(15)}`
    )
    console.log('─'.repeat(80))
    
    let totalSource = 0
    let totalTarget = 0
    let totalSkipped = 0
    
    for (const stat of stats) {
      console.log(
        `${stat.table.padEnd(30)} ${stat.sourceCount.toString().padStart(10)} ${stat.targetCount.toString().padStart(10)} ${stat.skippedCount.toString().padStart(10)} ${stat.duration.toString().padStart(15)}`
      )
      totalSource += stat.sourceCount
      totalTarget += stat.targetCount
      totalSkipped += stat.skippedCount
    }
    
    console.log('═'.repeat(80))
    console.log(
      `${'TOTAL'.padEnd(30)} ${totalSource.toString().padStart(10)} ${totalTarget.toString().padStart(10)} ${totalSkipped.toString().padStart(10)}`
    )
    console.log('═'.repeat(80))

    console.log('\n✅ Migration completed successfully!')
  } catch (error) {
    console.error('\n❌ Migration failed:', error)
    process.exit(1)
  } finally {
    await sourceDb.end()
    await targetDb.end()
  }
}

main()

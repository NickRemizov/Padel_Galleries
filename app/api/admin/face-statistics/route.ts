import { createClient } from "@/lib/supabase/server"
import { NextResponse } from "next/server"

export async function GET() {
  const supabase = await createClient()

  try {
    // Overall statistics
    const { data: overallStatsArray, error: overallError } = await supabase.rpc("get_face_statistics_overall")

    if (overallError) {
      console.error("[v0] RPC error:", overallError)
      // Fallback to direct query if RPC doesn't exist
      const { data: personStats } = await supabase
        .from("photo_faces")
        .select("person_id")
        .eq("verified", true)
        .not("person_id", "is", null)

      const personCounts =
        personStats?.reduce((acc: Record<string, number>, face) => {
          acc[face.person_id] = (acc[face.person_id] || 0) + 1
          return acc
        }, {}) || {}

      const faceCounts = Object.values(personCounts)
      const totalPeople = faceCounts.length
      const totalFaces = faceCounts.reduce((sum, count) => sum + count, 0)

      if (totalPeople === 0) {
        return NextResponse.json({
          overall: {
            total_people: 0,
            total_verified_faces: 0,
            avg_faces_per_person: "0.00",
            min_faces: 0,
            max_faces: 0,
          },
          distribution: [],
          top_people: [],
          histogram: [
            { face_range: "1 face", people_count: 0, total_faces: 0 },
            { face_range: "2 faces", people_count: 0, total_faces: 0 },
            { face_range: "3-4 faces", people_count: 0, total_faces: 0 },
            { face_range: "5-9 faces", people_count: 0, total_faces: 0 },
            { face_range: "10-14 faces", people_count: 0, total_faces: 0 },
            { face_range: "15-19 faces", people_count: 0, total_faces: 0 },
            { face_range: "20+ faces", people_count: 0, total_faces: 0 },
          ],
        })
      }

      const avgFaces = totalFaces / totalPeople

      // Distribution by thresholds
      const thresholds = [1, 3, 5, 10, 15, 20]
      const distribution = thresholds.map((threshold) => {
        const peopleCount = faceCounts.filter((count) => count >= threshold).length
        const totalFacesForThreshold = faceCounts
          .filter((count) => count >= threshold)
          .reduce((sum, count) => sum + count, 0)

        return {
          threshold: `>= ${threshold} faces`,
          people_count: peopleCount,
          total_faces: totalFacesForThreshold,
          percentage: ((peopleCount / totalPeople) * 100).toFixed(1),
        }
      })

      // Top people by face count
      const { data: topPeople } = await supabase
        .from("photo_faces")
        .select(`
          person_id,
          people!inner(real_name, telegram_name)
        `)
        .eq("verified", true)
        .not("person_id", "is", null)

      const peopleMap =
        topPeople?.reduce((acc: Record<string, any>, face) => {
          if (!acc[face.person_id]) {
            acc[face.person_id] = {
              person_id: face.person_id,
              real_name: face.people.real_name,
              telegram_name: face.people.telegram_name,
              count: 0,
            }
          }
          acc[face.person_id].count++
          return acc
        }, {}) || {}

      const topPeopleList = Object.values(peopleMap)
        .sort((a: any, b: any) => b.count - a.count)
        .slice(0, 20)

      // Histogram
      const histogram = [
        { range: "1 face", min: 1, max: 1 },
        { range: "2 faces", min: 2, max: 2 },
        { range: "3-4 faces", min: 3, max: 4 },
        { range: "5-9 faces", min: 5, max: 9 },
        { range: "10-14 faces", min: 10, max: 14 },
        { range: "15-19 faces", min: 15, max: 19 },
        { range: "20+ faces", min: 20, max: 999 },
      ].map((bucket) => {
        const peopleInBucket = faceCounts.filter((count) => count >= bucket.min && count <= bucket.max)
        return {
          face_range: bucket.range,
          people_count: peopleInBucket.length,
          total_faces: peopleInBucket.reduce((sum, count) => sum + count, 0),
        }
      })

      return NextResponse.json({
        overall: {
          total_people: totalPeople,
          total_verified_faces: totalFaces,
          avg_faces_per_person: avgFaces.toFixed(2),
          min_faces: Math.min(...faceCounts),
          max_faces: Math.max(...faceCounts),
        },
        distribution,
        top_people: topPeopleList,
        histogram,
      })
    }

    const overallStats = Array.isArray(overallStatsArray) && overallStatsArray.length > 0 ? overallStatsArray[0] : null

    if (!overallStats) {
      console.error("[v0] No statistics data returned from RPC")
      return NextResponse.json({
        overall: {
          total_people: 0,
          total_verified_faces: 0,
          avg_faces_per_person: "0.00",
          min_faces: 0,
          max_faces: 0,
        },
        distribution: [],
        histogram: [],
      })
    }

    const facesByCount = overallStats.faces_by_count || {}

    // Calculate distribution by thresholds
    const thresholds = [1, 3, 5, 10, 15, 20]
    const distribution = thresholds.map((threshold) => {
      let peopleCount = 0
      let totalFaces = 0

      // Sum up all ranges that meet the threshold
      Object.entries(facesByCount).forEach(([range, count]) => {
        const minFaces = range === "20+" ? 20 : Number.parseInt(range.split("-")[0])
        if (minFaces >= threshold) {
          peopleCount += count as number
          // Estimate total faces (use midpoint of range)
          const maxFaces = range === "20+" ? 25 : Number.parseInt(range.split("-")[1])
          const avgInRange = (minFaces + maxFaces) / 2
          totalFaces += (count as number) * avgInRange
        }
      })

      return {
        threshold: `>= ${threshold} faces`,
        people_count: peopleCount,
        total_faces: Math.round(totalFaces),
        percentage:
          overallStats.total_people > 0 ? ((peopleCount / overallStats.total_people) * 100).toFixed(1) : "0.0",
      }
    })

    // Create histogram from faces_by_count
    const histogram = [
      { range: "3-4 faces", key: "3-4" },
      { range: "5-9 faces", key: "5-9" },
      { range: "10-14 faces", key: "10-14" },
      { range: "15-19 faces", key: "15-19" },
      { range: "20+ faces", key: "20+" },
    ].map((bucket) => ({
      face_range: bucket.range,
      people_count: (facesByCount[bucket.key] as number) || 0,
      total_faces: 0, // We don't have exact total, but it's not critical for display
    }))

    return NextResponse.json({
      overall: {
        total_people: Number(overallStats.total_people) || 0,
        total_verified_faces: Number(overallStats.total_verified_faces) || 0,
        avg_faces_per_person: Number(overallStats.avg_faces_per_person || 0).toFixed(2),
        min_faces: 3, // Based on our query filter
        max_faces: Number(overallStats.total_verified_faces) || 0,
      },
      distribution,
      histogram,
    })
  } catch (error) {
    console.error("[v0] Error fetching face statistics:", error)
    return NextResponse.json({ error: "Failed to fetch statistics" }, { status: 500 })
  }
}

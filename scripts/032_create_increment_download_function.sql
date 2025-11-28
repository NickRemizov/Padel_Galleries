-- Create function to increment download count
CREATE OR REPLACE FUNCTION increment_download_count(image_id UUID)
RETURNS void
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
  UPDATE gallery_images
  SET download_count = download_count + 1
  WHERE id = image_id;
END;
$$;

-- Grant execute permission to authenticated users
GRANT EXECUTE ON FUNCTION increment_download_count(UUID) TO authenticated;
GRANT EXECUTE ON FUNCTION increment_download_count(UUID) TO anon;

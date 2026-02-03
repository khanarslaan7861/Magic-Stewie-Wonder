import java.io.IOException;
import java.io.InputStream;
import java.nio.file.AccessDeniedException;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.StandardCopyOption;
import java.nio.file.attribute.BasicFileAttributes;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;

public class Segregate {
    private static final Set<String> IMAGE_EXTS = new HashSet<>(Arrays.asList(
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".tif",
            ".tiff",
            ".webp",
            ".heic",
            ".heif",
            ".avif"
    ));

    private static final Set<String> VIDEO_EXTS = new HashSet<>(Arrays.asList(
            ".mp4",
            ".mov",
            ".m4v",
            ".avi",
            ".mkv",
            ".webm",
            ".wmv",
            ".flv",
            ".mpeg",
            ".mpg",
            ".3gp",
            ".3g2",
            ".mts",
            ".m2ts"
    ));

    private static final int HASH_CHUNK_SIZE = 8 * 1024 * 1024;

    public static void main(String[] args) throws IOException {
        Path root = Paths.get("").toAbsolutePath();
        Path resourcesDir = root.resolve("resources");
        Path segregatedDir = root.resolve("segregated");
        if (!Files.exists(resourcesDir)) {
            throw new IllegalStateException("resources folder not found: " + resourcesDir);
        }
        segregateResources(resourcesDir, segregatedDir);
    }

    private static void segregateResources(Path resourcesDir, Path segregatedDir) throws IOException {
        Path imagesDir = segregatedDir.resolve("images");
        Path videosDir = segregatedDir.resolve("videos");
        Path othersDir = segregatedDir.resolve("others");

        final long[] processed = {0};
        long progressEvery = 500;

        Map<Long, Object> seenSizes = new HashMap<>();

        Files.walkFileTree(resourcesDir, new SimpleFileVisitor<Path>() {
            @Override
            public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
                if (!attrs.isRegularFile()) {
                    return FileVisitResult.CONTINUE;
                }
                if (file.getFileName().toString().equals(".gitkeep")) {
                    return FileVisitResult.CONTINUE;
                }

                long size = attrs.size();
                Object entry = seenSizes.get(size);
                if (entry == null) {
                    seenSizes.put(size, file);
                } else if (entry instanceof Path) {
                    String existingDigest = fileDigest((Path) entry);
                    String digest = fileDigest(file);
                    if (digest.equals(existingDigest)) {
                        return FileVisitResult.CONTINUE;
                    }
                    Set<String> digests = new HashSet<>();
                    digests.add(existingDigest);
                    digests.add(digest);
                    seenSizes.put(size, digests);
                } else {
                    @SuppressWarnings("unchecked")
                    Set<String> digests = (Set<String>) entry;
                    String digest = fileDigest(file);
                    if (digests.contains(digest)) {
                        return FileVisitResult.CONTINUE;
                    }
                    digests.add(digest);
                }

                String ext = fileExtension(file);
                if (IMAGE_EXTS.contains(ext)) {
                    copyFile(file, imagesDir);
                } else if (VIDEO_EXTS.contains(ext)) {
                    copyFile(file, videosDir);
                } else {
                    copyFile(file, othersDir);
                }

                processed[0]++;
                processedProgress(progressEvery, processed[0]);
                return FileVisitResult.CONTINUE;
            }

            @Override
            public FileVisitResult visitFileFailed(Path file, IOException exc) {
                return FileVisitResult.CONTINUE;
            }
        });

        System.out.println();
    }

    private static String fileExtension(Path path) {
        String name = path.getFileName().toString();
        int idx = name.lastIndexOf('.');
        if (idx <= 0) {
            return "";
        }
        return name.substring(idx).toLowerCase(Locale.ROOT);
    }

    private static void copyFile(Path srcPath, Path destDir) throws IOException {
        Files.createDirectories(destDir);
        Path target = destDir.resolve(srcPath.getFileName().toString());
        if (Files.exists(target)) {
            String name = srcPath.getFileName().toString();
            int idx = name.lastIndexOf('.');
            String stem = idx > 0 ? name.substring(0, idx) : name;
            String suffix = idx > 0 ? name.substring(idx) : "";
            int i = 1;
            while (true) {
                Path candidate = destDir.resolve(stem + "_" + i + suffix);
                if (!Files.exists(candidate)) {
                    target = candidate;
                    break;
                }
                i++;
            }
        }
        try {
            Files.copy(srcPath, target, StandardCopyOption.COPY_ATTRIBUTES);
        } catch (AccessDeniedException e) {
            return;
        }
    }

    private static String fileDigest(Path path) throws IOException {
        MessageDigest digest;
        try {
            digest = MessageDigest.getInstance("MD5");
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("MD5 not available", e);
        }
        byte[] buffer = new byte[HASH_CHUNK_SIZE];
        try (InputStream in = Files.newInputStream(path)) {
            int read;
            while ((read = in.read(buffer)) != -1) {
                digest.update(buffer, 0, read);
            }
        }
        return toHex(digest.digest());
    }

    private static String toHex(byte[] bytes) {
        StringBuilder sb = new StringBuilder(bytes.length * 2);
        for (byte b : bytes) {
            sb.append(String.format("%02x", b));
        }
        return sb.toString();
    }

    private static void processedProgress(long progressEvery, long processed) {
        if (processed % progressEvery == 0) {
            System.out.print("\rProcessed " + processed + " files");
            System.out.flush();
        }
    }
}

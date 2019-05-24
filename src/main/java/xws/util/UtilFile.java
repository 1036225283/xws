package xws.util;


import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.List;

public class UtilFile {

    private static Logger log = LoggerFactory.getLogger(UtilFile.class);

    private FileOutputStream fileOutputStream;

    private FileInputStream fileInputStream;

    private File file;

    public UtilFile(String fileName) {
        // TODO Auto-generated constructor stub
        String strPath = fileName.substring(0, fileName.lastIndexOf(File.separator));

        File path = new File(strPath);
        if (!path.exists()) {
            boolean b = path.mkdirs();
            if (!b) {
                throw new RuntimeException("创建目录失败");
            }
        }

        file = new File(fileName);

        try {
            fileOutputStream = new FileOutputStream(file, true);
            fileInputStream = new FileInputStream(file);
        } catch (FileNotFoundException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }


    //读取文件通过文件名称
    @SuppressWarnings("resource")
    public static String readFile(String fileNameAndPath) {
        File file = new File(fileNameAndPath);

        FileInputStream input;
        byte[] bs = null;
        try {
            input = new FileInputStream(file);
            bs = new byte[(int) file.length()];
            input.read(bs);
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
            log.error("读取文件失败", e);
        }
        String fileContent = "";
        try {
            fileContent = new String(bs, "UTF-8");
            // fileContent = new String(fileContent.getBytes("UTF-8"), "UTF-8");
        } catch (UnsupportedEncodingException e) {
            // TODO Auto-generated catch block
            log.error("转换文件编码失败", e);
            e.printStackTrace();
        }
        return fileContent;
    }


    //从文件中读取字节数据
    public static byte[] readFileByte(String fileNameAndPath) {
        File file = new File(fileNameAndPath);
        try {
            FileInputStream input = new FileInputStream(file);
            byte[] bs = new byte[(int) file.length()];
            input.read(bs);
            input.close();
            return bs;
        } catch (Exception e) {
            // TODO Auto-generated catch block
            log.error("读取文件失败", e);
            e.printStackTrace();
            return new byte[0];
        }
    }

    //写文件
    public static void writeFile(String fileContent, String filePath) {
        FileOutputStream output;
        try {
            output = new FileOutputStream(new File(filePath));
            byte[] bs = fileContent.getBytes("UTF-8");
            output.write(bs);
            output.flush();
            output.close();
        } catch (Exception e) {
            // TODO Auto-generated catch block
            log.error("写文件失败", e);
        }

    }

    //追加string
    public void append(String value) {
        try {
            byte[] bs = value.getBytes("UTF-8");
            fileOutputStream.write(bs);
            fileOutputStream.flush();
        } catch (Exception e) {
            // TODO Auto-generated catch block
            log.error("写文件失败", e);
        }

    }


    //追加字节数据
    public void append(byte[] value) {
        try {
            fileOutputStream.write(value);
            fileOutputStream.flush();
        } catch (Exception e) {
            // TODO Auto-generated catch block
            log.error("写文件失败", e);
        }

    }

    /**
     * 创建目录
     *
     * @param path
     * @return
     */
    public static boolean mkdir(String path) {
        boolean result = false;
        File file = new File(path);
        if (!file.exists()) {
            result = file.mkdirs();
        }
        return result;
    }

    // 创建文件
    public static void mkfile(String fileName) {
        String path = fileName.substring(0, fileName.lastIndexOf(File.separator));

        File dir = new File(path);
        if (!dir.exists()) {
            dir.mkdirs();
        }

        try {
            File file = new File(fileName);
            if (file.exists()) {
                file.createNewFile();
            }
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }

    public void close() {
        try {
            if (fileOutputStream != null) {
                fileOutputStream.close();
            }

            if (fileInputStream != null) {
                fileInputStream.close();
            }

        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }

    /**
     * 两个文件内容叠加到一起，dest文件内容追加到src文件后面，返回新文件名
     *
     * @param srcFileName
     * @param destFileName
     * @return
     */
    public static String addToNew(String srcFileName, String destFileName) {
        String srcContent = UtilFile.readFile(srcFileName);
        String destContent = UtilFile.readFile(destFileName);
        srcContent = srcContent + destContent;
        String outFileName = srcFileName + "-addToNew.txt";
        UtilFile.writeFile(srcContent, outFileName);
        return outFileName;
    }

    public static void scanDirectory(File root, List<String> list) {
        File[] files = root.listFiles();
        for (File file : files) {
            if (file.isFile()) {
                String name = file.getAbsolutePath();
//				if (name.endsWith(".htm") || name.endsWith(".js")) {
//					if (name.contains("\\WebRoot\\js\\")) {
//						continue;
//					} else if (name.contains("\\js\\")) {
//						continue;
//					}
//					list.add(file.getAbsolutePath());
//				}
                list.add(file.getAbsolutePath());
            } else {
                scanDirectory(file, list);
            }
        }
    }

    //根据start和size读取文件
    public String read(int start, int length) {
        byte[] bs = new byte[1024];
        try {
            int size = fileInputStream.read(bs, start, length);
            return new String(bs, 0, size);
        } catch (Exception e) {
            System.out.println("文件读取异常");
            return null;
        }
    }

    //获取文件长度
    public int length() {
        return (int) file.length();
    }

    //获取指定目录下的所有文件名
    public static String[] getFileNames(String strPath) {
        File file = new File(strPath);
        if (file.isDirectory()) {
            File[] files = file.listFiles();
            String[] fileNames = new String[files.length];
            for (int i = 0; i < files.length; i++) {
                fileNames[i] = files[i].getName();
            }
            return fileNames;
        } else {
            throw new RuntimeException(strPath + "is not directory");
        }
    }


}

package xws.test;

import com.aparapi.Kernel;
import com.aparapi.ProfileInfo;
import com.aparapi.natives.NativeLoader;

import java.io.IOException;
import java.util.List;

/**
 * gpu test
 * Created by xws on 2019/4/1.
 */
public class GpuTest {

    public static void main(String[] args) {
        test0();
    }

    public static void test0() {
        try {
            NativeLoader.load();
            System.out.println("Aparapi JNI loaded successfully.");
        } catch (final IOException e) {
            System.out.println("Check your environment. Failed to load aparapi native library "
                    + " or possibly failed to locate opencl native library (opencl.dll/opencl.so)."
                    + " Ensure that OpenCL is in your PATH (windows) or in LD_LIBRARY_PATH (linux).");
        }
    }

    public static void test1() {
        final float result[] = new float[2048 * 2048];
        Kernel k = new Kernel() {
            public void run() {
                final int gid = getGlobalId();
                result[gid] = 0f;
            }
        };
        k.execute(result.length);
        List<ProfileInfo> profileInfo = k.getProfileInfo();

        for (final ProfileInfo p : profileInfo) {
            System.out.print(" " + p.getType() + " " + p.getLabel() + " " + (p.getStart() / 1000) + " .. "
                    + (p.getEnd() / 1000) + " " + ((p.getEnd() - p.getStart()) / 1000) + "us");
            System.out.println();
        }
        k.dispose();
    }
}

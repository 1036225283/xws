package xws.util;

import javax.xml.crypto.Data;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;

public class UtilDate {

    static SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
    static SimpleDateFormat yyyyMMddFormat = new SimpleDateFormat("yyyy-MM-dd");
    static SimpleDateFormat yyyyMMddFormatSlash = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");

    public static void main(String[] args) throws Exception {
//        System.out.println("周一" + getWeek("2019-04-08"));
//        System.out.println("周二" + getWeek("2019-04-09"));
//        System.out.println("周三" + getWeek("2019-04-10"));
//        System.out.println("周四" + getWeek("2019-04-11"));
//        System.out.println("周五" + getWeek("2019-04-12"));
//        System.out.println("周六" + getWeek("2019-04-13"));
//        System.out.println("周日" + getWeek("2019-04-14"));
//        System.out.println("周一" + getWeek("2019-04-15"));

//        System.out.println("周一" + getFriday("2019-04-08"));
//        System.out.println("周二" + getFriday("2019-04-09"));
//        System.out.println("周三" + getFriday("2019-04-10"));
//        System.out.println("周四" + getFriday("2019-04-11"));
//        System.out.println("周五" + getFriday("2019-04-12"));
//        System.out.println("周六" + getFriday("2019-04-13"));
//        System.out.println("周日" + getFriday("2019-04-14"));
//        System.out.println("周一" + getFriday("2019-04-15"));

        System.out.println("周一" + getMonday("2019-04-08"));
        System.out.println("周二" + getMonday("2019-04-09"));
        System.out.println("周三" + getMonday("2019-04-10"));
        System.out.println("周四" + getMonday("2019-04-11"));
        System.out.println("周五" + getMonday("2019-04-12"));
        System.out.println("周六" + getMonday("2019-04-13"));
        System.out.println("周日" + getMonday("2019-04-14"));
        System.out.println("周一" + getMonday("2019-04-15"));

    }

    /**
     * string to date
     *
     * @param date
     * @return
     */
    public static Date yyyyMMddHHmmssToDate(String date) {

        Date date2 = null;
        try {
            date2 = dateFormat.parse(date);
        } catch (ParseException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return date2;
    }

    public static Date yyyyMMddHHmmssSlashToDate(String date) {

        Date date2 = null;
        try {
            date2 = yyyyMMddFormatSlash.parse(date);
        } catch (ParseException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return date2;
    }

    /**
     * string to date
     *
     * @param date
     * @return
     */
    public static Date yyyyMMddToDate(String date) {
        Date date2 = null;
        try {
            date2 = yyyyMMddFormat.parse(date);
        } catch (ParseException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return date2;
    }

    /**
     * date to string
     *
     * @param date
     * @return
     */
    public static String dateToyyyyMMddHHmmss(Date date) {
        String strDate = dateFormat.format(date);
        return strDate;
    }

    public static String dateToyyyyMMddHHmmssSlash(Date date) {
        String strDate = yyyyMMddFormatSlash.format(date);
        return strDate;
    }

    /**
     * date to string
     *
     * @param date
     * @return
     */
    public static String dateToyyyyMMdd(Date date) {
        String strDate = yyyyMMddFormat.format(date);
        return strDate;
    }

    // yyyy-MM-dd to yyyy-MM
    public static String dateToyyyyMM(String date) {
        Date tmp = yyyyMMddToDate(date);
        SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM");
        String strDate = dateFormat.format(tmp);
        return strDate;
    }

    // 给定一个时间，对月份进行加减
    public static String dateAddMonth(String strDate, int num) {
        Date dtDate = yyyyMMddToDate(strDate);
        Calendar cal = Calendar.getInstance();
        cal.setTime(dtDate);
        cal.add(Calendar.MONTH, 1);
        return dateToyyyyMMdd(cal.getTime());
    }

    // 给定一个时间，对天数进行加减
    public static String dateAddDay(String strDate, int num) {
        Date dtDate = yyyyMMddToDate(strDate);
        Calendar cal = Calendar.getInstance();
        cal.setTime(dtDate);
        cal.add(Calendar.DAY_OF_MONTH, num);
        return dateToyyyyMMdd(cal.getTime());
    }

    // 根据日期，提供库名
    public static String getDataName(String strDate) {
        Date dtDate = yyyyMMddToDate(strDate);
        SimpleDateFormat dateFormat = new SimpleDateFormat("yyyyMM");
        return "Behavior" + dateFormat.format(dtDate);
    }

    // 根据日期，提供表名
    public static String getTableName(String strDate) {
        Date dtDate = yyyyMMddToDate(strDate);
        SimpleDateFormat dateFormat = new SimpleDateFormat("dd");
        return "Behavior" + dateFormat.format(dtDate);
    }

    // 获取月份
    public static int getMonth(String strDate) {
        Date dtDate = yyyyMMddToDate(strDate);
        Calendar cal = Calendar.getInstance();
        cal.setTime(dtDate);
        return cal.get(Calendar.MONTH);
    }

    //计算两个日期之间相差多少天
    public static long dateSubToDay(String strStartDate, String strEndDate) throws Exception {
        Date beginDate;
        Date endDate;
        beginDate = yyyyMMddFormat.parse(strStartDate);
        endDate = yyyyMMddFormat.parse(strEndDate);
        long day = (endDate.getTime() - beginDate.getTime()) / (24 * 60 * 60 * 1000);
        return day;
    }

    //根据日期判断是周一还是周五
    public static int getWeek(Date date) {
        Calendar cal = Calendar.getInstance();
        cal.setTime(date);
        return cal.get(Calendar.DAY_OF_WEEK);
    }

    //根据日期判断是周一还是周五
    public static int getWeek(String strDate) {
        Calendar cal = Calendar.getInstance();
        cal.setTime(yyyyMMddToDate(strDate));
        return cal.get(Calendar.DAY_OF_WEEK);
    }

    //判断是否周一
    public static boolean isMonday(String strDate) {
        int day = getWeek(strDate);
        if (day == 2) {
            return true;
        } else {
            return false;
        }
    }

    //判断是否周二
    public static boolean isTuesday(String strDate) {
        int day = getWeek(strDate);
        if (day == 3) {
            return true;
        } else {
            return false;
        }
    }

    //判断是否周三
    public static boolean isWednesday(String strDate) {
        int day = getWeek(strDate);
        if (day == 4) {
            return true;
        } else {
            return false;
        }
    }

    //判断是否周四
    public static boolean isThursday(String strDate) {
        int day = getWeek(strDate);
        if (day == 5) {
            return true;
        } else {
            return false;
        }
    }

    //判断是否周五
    public static boolean isFriday(String strDate) {
        int day = getWeek(strDate);
        if (day == 6) {
            return true;
        } else {
            return false;
        }
    }

    //判断是否周六
    public static boolean isSaturday(String strDate) {
        int day = getWeek(strDate);
        if (day == 7) {
            return true;
        } else {
            return false;
        }
    }

    //判断是否周日
    public static boolean isSunday(String strDate) {
        int day = getWeek(strDate);
        if (day == 1) {
            return true;
        } else {
            return false;
        }
    }

    //获取给定日期的周一
    public static String getMonday(String strDate) {
        int day = getWeek(strDate);
        if (day == 1) {
            return UtilDate.dateAddDay(strDate, -6);
        } else {
            return UtilDate.dateAddDay(strDate, -(day - 2));
        }
    }

    //获取给定日期的周五
    public static String getFriday(String strDate) {
        int day = getWeek(strDate);

        if (day == 2 || day == 3 || day == 4 || day == 5 || day == 6) {
            return UtilDate.dateAddDay(strDate, 6 - day);
        }

        if (day == 7) {
            return UtilDate.dateAddDay(strDate, -1);
        }

        return UtilDate.dateAddDay(strDate, -2);
    }

}

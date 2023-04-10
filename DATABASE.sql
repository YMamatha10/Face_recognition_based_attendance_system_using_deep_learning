/*
SQLyog Enterprise - MySQL GUI v6.56
MySQL - 5.5.5-10.1.13-MariaDB : Database - smart_attendance
*********************************************************************
*/

/*!40101 SET NAMES utf8 */;

/*!40101 SET SQL_MODE=''*/;

/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;

CREATE DATABASE /*!32312 IF NOT EXISTS*/`smart_attendance` /*!40100 DEFAULT CHARACTER SET latin1 */;

USE `smart_attendance`;

/*Table structure for table `attendance` */

DROP TABLE IF EXISTS `attendance`;

CREATE TABLE `attendance` (
  `id` int(10) NOT NULL AUTO_INCREMENT,
  `rno` varchar(100) DEFAULT NULL,
  `in_time` varchar(100) DEFAULT NULL,
  `in_status` varchar(100) DEFAULT NULL,
  `out_time` varchar(100) DEFAULT NULL,
  `out_status` varchar(100) DEFAULT NULL,
  `overall_time` varchar(100) DEFAULT NULL,
  `date1` varchar(100) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=6 DEFAULT CHARSET=latin1;

/*Data for the table `attendance` */

insert  into `attendance`(`id`,`rno`,`in_time`,`in_status`,`out_time`,`out_status`,`overall_time`,`date1`) values (4,'16HP1A0506','10:01','Early Coming',NULL,NULL,NULL,'2021-12-30'),(5,'0427','10:01','Early Coming','10:01','Early Out','0:0','2021-12-30');

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;

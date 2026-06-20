CREATE TABLE `conversions` (
	`id` int AUTO_INCREMENT NOT NULL,
	`userId` int NOT NULL,
	`sourceFileName` varchar(255) NOT NULL,
	`sourceFormat` varchar(50) NOT NULL,
	`targetFormat` varchar(50) NOT NULL,
	`status` enum('pending','converting','completed','failed') NOT NULL DEFAULT 'pending',
	`errorMessage` text,
	`sourceFileKey` varchar(255) NOT NULL,
	`sourceFileUrl` varchar(512) NOT NULL,
	`resultFileKey` varchar(255),
	`resultFileUrl` varchar(512),
	`resultFileName` varchar(255),
	`fileSize` int,
	`resultFileSize` int,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	`updatedAt` timestamp NOT NULL DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	`completedAt` timestamp,
	CONSTRAINT `conversions_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
ALTER TABLE `conversions` ADD CONSTRAINT `conversions_userId_users_id_fk` FOREIGN KEY (`userId`) REFERENCES `users`(`id`) ON DELETE no action ON UPDATE no action;